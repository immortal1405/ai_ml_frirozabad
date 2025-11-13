# Transformer Models Training & Inference

## Overview

This document describes the **Transformer architecture** implementation for PM2.5 air quality prediction in Firozabad. Two Transformer variants have been implemented:

1. **Standard Transformer** - Pure attention-based model with positional encoding
2. **Enhanced Transformer** - Transformer + Physics-Informed features

---

## Architecture Details

### Standard Transformer (`TransformerPredictor`)

**Components:**
- **Input Projection**: Linear layer to project 130 features → 256 dimensions
- **Positional Encoding**: Sine/cosine encoding for temporal information
- **Transformer Encoder**: 6 layers, 8 attention heads, 1024 FFN dimension
- **CLS Token**: Learnable token for sequence representation
- **Output Network**: 3-layer MLP with GELU activation

**Key Features:**
- Pre-LN Transformer (more stable training)
- Multi-head self-attention (8 heads)
- GELU activation (better for Transformers)
- Global pooling via CLS token
- Dropout: 0.3

**Parameters**: ~3.5M

### Enhanced Transformer (`EnhancedTransformerPredictor`)

**Additional Components:**
- **Physics-Informed Layer**: Processes meteorological features
  - Wind effects (advection)
  - PBL height (dispersion)
  - Humidity (hygroscopic growth)
  - Temperature (stability)
  - Diurnal patterns
- **Fusion Layer**: Combines Transformer output with physics features

**Parameters**: ~4.2M

---

## Training Configuration

### Hyperparameters

```python
d_model = 256              # Embedding dimension
nhead = 8                  # Number of attention heads
num_layers = 6             # Transformer encoder layers
dim_feedforward = 1024     # FFN hidden dimension
dropout = 0.3              # Dropout rate
batch_size = 128/64        # 128 for standard, 64 for enhanced
learning_rate = 0.0001     # Lower than RNN models
betas = (0.9, 0.98)       # AdamW betas (Transformer standard)
weight_decay = 1e-5
epochs = 30                # More epochs needed
patience = 10              # Early stopping patience
```

### Why Different from RNN Models?

| Aspect | RNN Models | Transformer Models |
|--------|------------|-------------------|
| Learning Rate | 0.001 | 0.0001 (10x lower) |
| Epochs | 20 | 30 (50% more) |
| Patience | 7 | 10 (more lenient) |
| Batch Size | 256/128 | 128/64 (smaller) |
| Optimizer Betas | (0.9, 0.999) | (0.9, 0.98) |
| Warmup | No | Yes (optional) |

**Reasons:**
- Transformers are more parameter-efficient but need more careful tuning
- Attention mechanism is memory-intensive → smaller batches
- No recurrence → needs more epochs to learn temporal patterns
- Lower LR prevents attention collapse

---

## File Structure

```
aiml_firozabad/
├── model_arch.py                          # Contains Transformer architectures
│   ├── PositionalEncoding                 # Sin/cos positional encoding
│   ├── TransformerPredictor              # Standard Transformer
│   └── EnhancedTransformerPredictor      # Transformer + Physics
│
├── modal_training_transformer.py          # Training script for Transformers
│   └── train_transformer()               # Train function
│
├── modal_inferencing_transformer.py       # Inference script for Transformers
│   └── run_transformer_inference()       # Inference function
│
└── TRANSFORMER_GUIDE.md                  # This file
```

---

## Usage

### 1. Training Transformer Models

```bash
# Train both Transformer models (Standard + Enhanced)
modal run modal_training_transformer.py
```

**What happens:**
- Loads `firozabad_sequences_hourly_complete.pkl` from Modal volume
- Trains Standard Transformer (30 epochs, batch 128)
- Trains Enhanced Transformer (30 epochs, batch 64)
- Saves best checkpoints to `checkpoints_firozabad/`
- Sequential execution (memory-intensive models)

**Estimated Time:**
- Standard Transformer: ~25-35 minutes
- Enhanced Transformer: ~30-40 minutes
- **Total: ~60-75 minutes**

**Output Files:**
- `firozabad_Transformer_best.pth`
- `firozabad_Transformer_checkpoint.pth`
- `firozabad_Transformer_results.json`
- `firozabad_Enhanced_Transformer_best.pth`
- `firozabad_Enhanced_Transformer_checkpoint.pth`
- `firozabad_Enhanced_Transformer_results.json`

### 2. Running Inference

```bash
# Run inference for Transformer models
modal run modal_inferencing_transformer.py
```

**Requirements:**
- Trained Transformer models in Modal volume
- `cleaned_pm25_firozabad_inference_nov9_11.csv` uploaded

**Output:**
- `firozabad_transformer_inference_nov9_11_2025.json`
- `transformer_metrics_summary_nov9_11_2025.json`

### 3. Download Results

```bash
# Download full results
modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_transformer_inference_nov9_11_2025.json .

# Download summary only
modal volume get ai_ml_firozabad inference_results_firozabad/transformer_metrics_summary_nov9_11_2025.json .
```

---

## Expected Performance

### Compared to Other Models

| Model | RMSE (Expected) | Parameters | Training Time | Inference Speed |
|-------|----------------|------------|---------------|-----------------|
| BiGRU | 5.97 | 2.1M | 12 min | Fast |
| GRU | 6.02 | 2.1M | 12 min | Fast |
| LSTM | 6.53 | 2.8M | 15 min | Moderate |
| **Transformer** | **6.2-7.5** | **3.5M** | **30 min** | **Moderate** |
| **Enhanced Transformer** | **6.5-8.0** | **4.2M** | **35 min** | **Slow** |
| Enhanced BiLSTM | 7.09 | 3.5M | 18 min | Slow |
| XGBoost | 8.85 | N/A | 25 min | Very Fast |

### Transformer Advantages

✅ **Pure attention mechanism** - no recurrence bias
✅ **Parallelizable training** - faster than sequential RNNs in theory
✅ **Long-range dependencies** - O(1) attention distance
✅ **State-of-the-art in NLP** - proven architecture
✅ **Interpretable attention** - can visualize what model attends to

### Transformer Challenges

❌ **Higher memory usage** - O(n²) attention complexity
❌ **Needs more data** - typically requires larger datasets
❌ **More epochs needed** - no recurrence for temporal patterns
❌ **Sensitive hyperparameters** - LR, warmup, dropout critical
❌ **May not beat RNNs** - for short sequences, RNNs often win

---

## Architecture Comparison

### Transformer vs LSTM/GRU

| Aspect | RNN (LSTM/GRU) | Transformer |
|--------|----------------|-------------|
| **Temporal Modeling** | Recurrent (sequential) | Attention (parallel) |
| **Memory** | Hidden state | Key-value pairs |
| **Complexity** | O(n) per step | O(n²) attention |
| **Parallelization** | Limited (sequential) | Full (all positions) |
| **Long-term Deps** | Vanishing gradient risk | Direct attention |
| **Best For** | Short-medium sequences | Long sequences |
| **Training Speed** | Slower (sequential) | Faster (parallel) |
| **Inference Speed** | Fast | Moderate |

### When to Use Each

**Use Transformer when:**
- You have long sequences (>200 timesteps)
- You need interpretability (attention weights)
- You want SOTA architecture
- You have compute budget for larger models

**Use RNN (GRU/LSTM) when:**
- Sequences are short-medium (<200 timesteps)
- Memory is limited
- You need fast inference
- You want proven performance on time series

**For PM2.5 Prediction (168 timesteps):**
→ **RNNs are competitive**, but Transformers worth trying for comparison

---

## Troubleshooting

### Out of Memory (OOM)

**Problem:** CUDA OOM during training

**Solutions:**
```python
# Reduce batch size
batch_size = 32  # Instead of 64/128

# Reduce model size
d_model = 128           # Instead of 256
num_layers = 4          # Instead of 6
dim_feedforward = 512   # Instead of 1024

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint(...)
```

### Training Not Converging

**Problem:** Val loss not decreasing

**Solutions:**
```python
# Lower learning rate
lr = 0.00005  # Instead of 0.0001

# Add warmup
warmup_steps = 1000
scheduler = get_linear_schedule_with_warmup(...)

# Increase patience
patience = 15  # Instead of 10

# Check for NaN gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Slow Training

**Problem:** Each epoch takes too long

**Solutions:**
```bash
# Use smaller batch size (counterintuitive but helps GPU utilization)
batch_size = 64

# Reduce sequence length (if possible)
lookback = 120  # Instead of 168

# Enable mixed precision (FP16)
torch.cuda.amp.autocast()

# Use Flash Attention (requires PyTorch 2.0+)
torch.nn.functional.scaled_dot_product_attention(..., enable_flash=True)
```

---

## Advanced Features

### 1. Attention Visualization

```python
# During inference, capture attention weights
outputs, attention_weights = model(sequences)

# attention_weights: (batch, num_heads, seq_len, seq_len)
# Visualize which timesteps the model attends to

import matplotlib.pyplot as plt
import seaborn as sns

# Average across heads
avg_attn = attention_weights[0].mean(dim=0).cpu().numpy()

plt.figure(figsize=(12, 10))
sns.heatmap(avg_attn, cmap='viridis')
plt.title('Transformer Attention Weights')
plt.xlabel('Key Position (Hours)')
plt.ylabel('Query Position (Hours)')
plt.savefig('attention_heatmap.png')
```

### 2. Learning Rate Warmup

```python
def get_lr_multiplier(step, warmup_steps=1000, d_model=256):
    """Transformer LR schedule from 'Attention is All You Need'"""
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

# Apply in training loop
for step, (sequences, targets) in enumerate(train_loader):
    lr = base_lr * get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### 3. Flash Attention (for H100)

```python
# Enable Flash Attention for 2-3x speedup
import torch.nn.functional as F

# In attention computation
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=dropout if training else 0.0,
    is_causal=False,
    enable_flash=True  # H100 optimization
)
```

---

## Integration with Existing Pipeline

### Update Visualization Script

```python
# Add Transformer models to MODELS list in visualization.py
MODELS = ['RandomForest', 'XGBoost', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 
          'Enhanced_BiLSTM', 'Enhanced_BiGRU', 
          'Transformer', 'Enhanced_Transformer']  # Add these

# Merge results from both inference runs
import json

# Load original results
with open('inference_results_firozabad/firozabad_inference_nov9_11_2025.json', 'r') as f:
    original_results = json.load(f)

# Load Transformer results
with open('inference_results_firozabad/firozabad_transformer_inference_nov9_11_2025.json', 'r') as f:
    transformer_results = json.load(f)

# Merge
all_results = {**original_results, **transformer_results}

# Save merged results
with open('inference_results_firozabad/firozabad_complete_inference_nov9_11_2025.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

### Update Report

Add Transformer results to `REPORT.md`:

```markdown
| Model | RMSE | MAE | R² | Parameters | Training Time |
|-------|------|-----|-----|-----------|---------------|
| BiGRU | 5.97 | 3.99 | 0.9766 | 2.1M | 12 min |
| Transformer | 6.XX | X.XX | 0.9XXX | 3.5M | 30 min |
| Enhanced Transformer | X.XX | X.XX | 0.9XXX | 4.2M | 35 min |
```

---

## References

**Original Papers:**
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Pre-LN Transformer (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745)

**Time Series with Transformers:**
- [Informer (Zhou et al., 2021)](https://arxiv.org/abs/2012.07436)
- [Autoformer (Wu et al., 2021)](https://arxiv.org/abs/2106.13008)
- [FEDformer (Zhou et al., 2022)](https://arxiv.org/abs/2201.12740)

---

## Summary

✅ **2 Transformer models added**: Standard + Enhanced
✅ **Separate training script**: `modal_training_transformer.py`
✅ **Separate inference script**: `modal_inferencing_transformer.py`
✅ **Same file structure**: Saves to same `checkpoints_firozabad/` directory
✅ **Production ready**: Can run independently after main training
✅ **Easy integration**: Results compatible with existing visualization pipeline

**Next Steps:**
1. Train Transformers: `modal run modal_training_transformer.py`
2. Run inference: `modal run modal_inferencing_transformer.py`
3. Merge results with existing 8 models
4. Update visualization and report with Transformer performance
5. Compare: Does pure attention beat recurrent models? 🤔

---

**Questions?** The Transformer models are now ready to train! 🚀
