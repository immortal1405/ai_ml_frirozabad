"""
Step 7: Firozabad PM2.5 Visualization & Analysis (Run Locally)
Downloads from Modal volume and creates comprehensive visualizations
for Nov 9-11, 2025 test period
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

CITY = 'firozabad'
MODELS = ['RandomForest', 'XGBoost', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 
          'Enhanced_BiLSTM', 'Enhanced_BiGRU']

def create_comprehensive_plot(city, model_type, predictions, targets, time_index, metrics):
    """Create comprehensive hourly analysis with diurnal patterns"""
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    time_index = pd.to_datetime(time_index)
    hours = time_index.hour.values
    
    # 1. Full Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_index, targets, label='Actual PM2.5', alpha=0.7, linewidth=1.5, color='#2E86AB')
    ax1.plot(time_index, predictions, label='Predicted PM2.5', alpha=0.7, linewidth=1.5, color='#A23B72')
    ax1.fill_between(time_index, predictions, targets, alpha=0.2, color='gray', label='Error')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{city.upper()} - {model_type}: Nov 9-11, 2025 Hourly Prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Detailed View (First 500 hours or all if less)
    ax2 = fig.add_subplot(gs[1, :2])
    plot_points = min(100, len(predictions))  # 72-100 hours for 3-day test
    ax2.plot(range(plot_points), targets[:plot_points], label='Actual', linewidth=2,
            marker='o', markersize=3, alpha=0.7, color='#2E86AB')
    ax2.plot(range(plot_points), predictions[:plot_points], label='Predicted', linewidth=2,
            marker='s', markersize=3, alpha=0.7, color='#A23B72')
    ax2.set_xlabel('Time Steps (Hours)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Detailed View (First {plot_points} Hours)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter Plot
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.scatter(targets, predictions, alpha=0.5, s=20, color='#F18F01')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    ax3.plot(targets, p(targets), 'g-', linewidth=2, alpha=0.7, label='Best fit')
    ax3.set_xlabel('Actual PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Scatter Plot (R²={metrics["R2"]:.4f})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Diurnal Cycle
    ax4 = fig.add_subplot(gs[2, 0])
    diurnal_actual = np.zeros(24)
    diurnal_pred = np.zeros(24)
    counts = np.zeros(24)
    
    for i, h in enumerate(hours):
        diurnal_actual[h] += targets[i]
        diurnal_pred[h] += predictions[i]
        counts[h] += 1
    
    diurnal_actual = np.divide(diurnal_actual, counts, where=counts>0, out=np.zeros_like(diurnal_actual))
    diurnal_pred = np.divide(diurnal_pred, counts, where=counts>0, out=np.zeros_like(diurnal_pred))
    
    hours_axis = np.arange(24)
    ax4.plot(hours_axis, diurnal_actual, marker='o', label='Actual', linewidth=2.5, color='#2E86AB')
    ax4.plot(hours_axis, diurnal_pred, marker='s', label='Predicted', linewidth=2.5, color='#A23B72')
    
    # Mark key periods
    ax4.axvspan(7, 10, alpha=0.1, color='red', label='Morning Peak')
    ax4.axvspan(14, 16, alpha=0.1, color='green', label='Afternoon Min')
    ax4.axvspan(18, 20, alpha=0.1, color='orange', label='Evening Peak')
    ax4.axvspan(22, 24, alpha=0.1, color='purple', label='Night High')
    ax4.axvspan(0, 6, alpha=0.1, color='purple')
    
    ax4.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax4.set_title('Diurnal Cycle (24-Hour Pattern)', fontsize=12, fontweight='bold')
    ax4.set_xticks(hours_axis)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Day vs Night
    ax5 = fig.add_subplot(gs[2, 1])
    day_mask = (hours >= 6) & (hours < 18)
    night_mask = (hours >= 18) | (hours < 6)
    
    bp = ax5.boxplot([targets[day_mask], predictions[day_mask],
                       targets[night_mask], predictions[night_mask]],
                      labels=['Day Actual', 'Day Pred', 'Night Actual', 'Night Pred'],
                      patch_artist=True)
    
    colors = ['#2E86AB', '#A23B72', '#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax5.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
    ax5.set_title('Day vs Night PM2.5', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Metrics Table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    metrics_text = f"""PERFORMANCE METRICS
{'='*35}

MSE:     {metrics['MSE']:.4f}
RMSE:    {metrics['RMSE']:.4f}
MAE:     {metrics['MAE']:.4f}
R²:      {metrics['R2']:.4f}
MAPE:    {metrics['MAPE']:.2f}%

VARIANCE ANALYSIS
{'='*35}

Actual:    {metrics['Variance_Actual']:.4f}
Predicted: {metrics['Variance_Pred']:.4f}
Diff:      {metrics['Variance_Diff']:.4f}

Match: {(1-metrics['Variance_Diff']/max(metrics['Variance_Actual'],1e-6))*100:.2f}%"""
    
    ax6.text(0.05, 0.95, metrics_text, fontsize=9, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{city.upper()} - {model_type}: Nov 9-11, 2025 PM2.5 Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    
    os.makedirs('results_firozabad', exist_ok=True)
    plt.savefig(f'results_firozabad/{city}_{model_type}_nov9_11_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results_firozabad/variance_comparison_nov9_11.png")

def create_summary_table():
    """Create summary performance table for Firozabad"""
    
    with open('inference_results_firozabad/firozabad_inference_nov9_11_2025.json', 'r') as f:
        results = json.load(f)
    
    summary_data = []
    for model in MODELS:
        if model in results and results[model] is not None:
            m = results[model]['metrics']
            summary_data.append({
                'Model': model,
                'RMSE': m['RMSE'],
                'MAE': m['MAE'],
                'R2': m['R2'],
                'MAPE': m['MAPE'],
                'Variance_Diff': m['Variance_Diff'],
                'Num_Samples': results[model].get('num_samples', 'N/A')
            })
        else:
            summary_data.append({
                'Model': model,
                'RMSE': 'N/A',
                'MAE': 'N/A',
                'R2': 'N/A',
                'MAPE': 'N/A',
                'Variance_Diff': 'N/A',
                'Num_Samples': 'N/A'
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results_firozabad/performance_summary_nov9_11.csv', index=False)
    
    print("\n" + "="*90)
    print("FIROZABAD PM2.5 PREDICTION - PERFORMANCE SUMMARY (NOV 9-11, 2025)")
    print("="*90 + "\n")
    print(summary_df.to_string(index=False))
    print("\n✓ Saved: results_firozabad/performance_summary_nov9_11.csv")
    print()
    
    # Find best model
    valid_results = [r for r in summary_data if r['RMSE'] != 'N/A']
    if valid_results:
        best_rmse = min(valid_results, key=lambda x: x['RMSE'])
        best_r2 = max(valid_results, key=lambda x: x['R2'])
        
        print("="*90)
        print("BEST PERFORMING MODELS")
        print("="*90)
        print(f"Lowest RMSE: {best_rmse['Model']} (RMSE={best_rmse['RMSE']:.4f}, R²={best_rmse['R2']:.4f})")
        print(f"Highest R²:  {best_r2['Model']} (R²={best_r2['R2']:.4f}, RMSE={best_r2['RMSE']:.4f})")
        print("="*90 + "\n")

def create_model_comparison():
    """Create comprehensive comparison across all 8 models for Firozabad"""

    with open('inference_results_firozabad/firozabad_inference_nov9_11_2025.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Firozabad PM2.5 - Model Comparison (Nov 9-11, 2025)',
                fontsize=16, fontweight='bold')
    
    models = MODELS
    metrics_names = ['RMSE', 'MAE', 'R2', 'MAPE']
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        values = []
        colors = []
        for model in models:
            if model in results and results[model] is not None:
                values.append(results[model]['metrics'][metric])
                # Color ML models differently from DL models
                if model in ['RandomForest', 'XGBoost']:
                    colors.append('#2E86AB')  # Blue for ML
                elif 'Enhanced' in model:
                    colors.append('#A23B72')  # Purple for Enhanced
                else:
                    colors.append('#F18F01')  # Orange for standard DL
            else:
                values.append(0)
                colors.append('#CCCCCC')  # Gray for missing
        
        bars = ax.bar(range(len(models)), values, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='ML Models', alpha=0.8),
        Patch(facecolor='#F18F01', label='Standard DL', alpha=0.8),
        Patch(facecolor='#A23B72', label='Enhanced DL', alpha=0.8)
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results_firozabad/model_comparison_nov9_11.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results_firozabad/model_comparison_nov9_11.png")

def create_variance_comparison():
    """Create variance comparison for Firozabad across all models"""

    with open('inference_results_firozabad/firozabad_inference_nov9_11_2025.json', 'r') as f:
        results = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Firozabad PM2.5 - Variance Analysis (Nov 9-11, 2025)',
                fontsize=16, fontweight='bold')
    
    models = MODELS
    x = np.arange(len(models))
    width = 0.35
    
    actual_vars = []
    pred_vars = []
    
    for model in models:
        if model in results and results[model] is not None:
            actual_vars.append(results[model]['metrics']['Variance_Actual'])
            pred_vars.append(results[model]['metrics']['Variance_Pred'])
        else:
            actual_vars.append(0)
            pred_vars.append(0)
    
    # Variance comparison bar chart
    bars1 = ax1.bar(x - width/2, actual_vars, width, label='Actual Variance',
                   color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax1.bar(x + width/2, pred_vars, width, label='Predicted Variance',
                   color='#A23B72', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax1.set_title('Variance: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Variance difference
    var_diffs = []
    for model in models:
        if model in results and results[model] is not None:
            var_diffs.append(results[model]['metrics']['Variance_Diff'])
        else:
            var_diffs.append(0)
    
    colors = ['#2E86AB' if m in ['RandomForest', 'XGBoost'] 
              else '#A23B72' if 'Enhanced' in m 
              else '#F18F01' for m in models]
    
    bars3 = ax2.bar(range(len(models)), var_diffs, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance Difference', fontsize=12, fontweight='bold')
    ax2.set_title('Variance Difference (|Actual - Predicted|)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, var_diffs):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results_firozabad/variance_comparison_nov9_11.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: results_firozabad/variance_comparison_nov9_11.png")

def main():
    print("="*90)
    print("STEP 7: FIROZABAD PM2.5 VISUALIZATION & ANALYSIS (NOV 9-11, 2025)")
    print("="*90 + "\n")

    if not os.path.exists('inference_results_firozabad/firozabad_inference_nov9_11_2025.json'):
        print("ERROR: firozabad_inference_nov9_11_2025.json not found!")
        print("\nPlease download from Modal volume first:")
        print("  modal volume get ai_ml_firozabad inference_results_firozabad/ .")
        print("\nOr download just the results file:")
        print("  modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_inference_nov9_11_2025.json .")
        return
    
    os.makedirs('results_firozabad', exist_ok=True)

    with open('inference_results_firozabad/firozabad_inference_nov9_11_2025.json', 'r') as f:
        results = json.load(f)
    
    # Create individual model plots
    print("Creating visualizations for Firozabad...\n")
    for model in MODELS:
        if model in results and results[model] is not None:
            data = results[model]
            create_comprehensive_plot(
                data['city'],
                data['model_type'],
                data['predictions'],
                data['targets'],
                data['time_index'],
                data['metrics']
            )
        else:
            print(f"⚠ Skipping {model} - no results available")
    
    # Create comparison plots
    print("\nCreating comparison visualizations...")
    create_model_comparison()
    create_variance_comparison()
    
    # Create summary table
    create_summary_table()
    
    print("\n" + "="*90)
    print("✓ VISUALIZATION COMPLETE")
    print("="*90)
    print("\nGenerated files in 'results_firozabad/' folder:")
    print(f"  • firozabad_{{model}}_nov9_11_analysis.png - Individual model analysis (8 files)")
    print("  • model_comparison_nov9_11.png - Metrics comparison across all models")
    print("  • variance_comparison_nov9_11.png - Variance analysis")
    print("  • performance_summary_nov9_11.csv - Complete metrics table")
    print("\nTest Period: November 9-11, 2025 (72 hours)")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
