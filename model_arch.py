"""
Complete Model Architectures for Hourly PM2.5 Prediction
- Enhanced Bi-LSTM with Attention + Physics-Informed Diffusion
- Enhanced Bi-GRU with Attention + Physics-Informed Diffusion
- Standard LSTM
- Standard GRU
- Standard BiLSTM
- Standard BiGRU
- RandomForest
- XGBoost
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_size, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.fc_out(context)
        
        return output, attention_weights

class DiurnalAwarePhysicsLayer(nn.Module):
    """Physics-informed diffusion layer with diurnal awareness"""
    def __init__(self, input_size, hidden_size):
        super(DiurnalAwarePhysicsLayer, self).__init__()
        
        # Learnable physics parameters
        self.diffusion_coeff = nn.Parameter(torch.ones(1) * 0.5)
        self.advection_coeff = nn.Parameter(torch.ones(1) * 0.5)
        self.deposition_coeff = nn.Parameter(torch.ones(1) * 0.1)
        self.pbl_effect = nn.Parameter(torch.ones(1) * 0.3)
        
        # Day-Night specific processing
        self.day_night_processor = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Wind processing (advection)
        self.wind_processor = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # PBL proxy (dispersion)
        self.pbl_processor = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Humidity (hygroscopic growth)
        self.humidity_processor = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Temperature (dispersion)
        self.temp_processor = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, features):
        """
        features: dict with:
        - 'is_night': (batch, 1) binary
        - 'hour_norm': (batch, 1) 0-1
        - 'wind': (batch, 2) wind_u, wind_v
        - 'pbl_proxy': (batch, 1)
        - 'humidity': (batch, 1)
        - 'temperature': (batch, 1)
        """
        
        # Day-night effect
        day_night = torch.cat([features['is_night'], features['hour_norm']], dim=1)
        day_night_effect = self.day_night_processor(day_night)
        
        # Physics components
        wind_effect = self.wind_processor(features['wind'])
        pbl_effect = self.pbl_processor(features['pbl_proxy'])
        humidity_effect = self.humidity_processor(features['humidity'])
        temp_effect = self.temp_processor(features['temperature'])
        
        # Apply learnable coefficients
        diffusion = torch.sigmoid(self.diffusion_coeff)
        advection = torch.tanh(self.advection_coeff)
        deposition = torch.sigmoid(self.deposition_coeff)
        pbl = torch.sigmoid(self.pbl_effect)
        
        # Combine all effects
        combined = torch.cat([
            advection * wind_effect,
            pbl * pbl_effect,
            deposition * humidity_effect,
            diffusion * temp_effect,
            day_night_effect
        ], dim=1)
        
        physics_output = self.fusion(combined)
        
        return physics_output

class EnhancedBiLSTMAttentionComplete(nn.Module):
    """BiLSTM with Multi-Head Attention + Physics-Informed Diffusion"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, num_heads=8):
        super(EnhancedBiLSTMAttentionComplete, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project bidirectional to single
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        
        # Physics layer
        self.physics_layer = DiurnalAwarePhysicsLayer(input_size, hidden_size)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x, physics_features):
        # x: (batch, seq_len, input_size)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.projection(lstm_out)
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out)
        lstm_context = torch.mean(lstm_out + attn_out, dim=1)
        
        # Physics
        physics_out = self.physics_layer(physics_features)
        
        # Fusion
        combined = torch.cat([lstm_context, physics_out], dim=1)
        fused = self.fusion(combined)
        
        # Output
        output = self.output_net(fused)
        
        return output, attn_weights

class EnhancedBiGRUAttentionComplete(nn.Module):
    """BiGRU with Multi-Head Attention + Physics-Informed Diffusion"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, num_heads=8):
        super(EnhancedBiGRUAttentionComplete, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Attention
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        
        # Physics
        self.physics_layer = DiurnalAwarePhysicsLayer(input_size, hidden_size)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x, physics_features):
        gru_out, _ = self.bigru(x)
        gru_out = self.projection(gru_out)
        
        attn_out, attn_weights = self.attention(gru_out)
        gru_context = torch.mean(gru_out + attn_out, dim=1)
        
        physics_out = self.physics_layer(physics_features)
        
        combined = torch.cat([gru_context, physics_out], dim=1)
        fused = self.fusion(combined)
        
        output = self.output_net(fused)
        
        return output, attn_weights

# ============================================================================
# STANDARD MODELS FOR COMPARISON
# ============================================================================

class StandardLSTM(nn.Module):
    """Standard LSTM without attention or physics"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(StandardLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x, physics_features=None):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.output_net(last_output)
        return output, None

class StandardGRU(nn.Module):
    """Standard GRU without attention or physics"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(StandardGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x, physics_features=None):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.output_net(last_output)
        return output, None

class StandardBiLSTM(nn.Module):
    """Standard Bidirectional LSTM without attention or physics"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(StandardBiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x, physics_features=None):
        bilstm_out, _ = self.bilstm(x)
        last_output = bilstm_out[:, -1, :]
        output = self.output_net(last_output)
        return output, None

class StandardBiGRU(nn.Module):
    """Standard Bidirectional GRU without attention or physics"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(StandardBiGRU, self).__init__()
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x, physics_features=None):
        bigru_out, _ = self.bigru(x)
        last_output = bigru_out[:, -1, :]
        output = self.output_net(last_output)
        return output, None

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    """Transformer encoder for time series prediction"""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.3):
        super(TransformerPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN Transformer (more stable)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Global pooling strategies
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x, physics_features=None):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Extract representation
        if self.use_cls_token:
            # Use CLS token representation
            representation = transformer_out[:, 0, :]
        else:
            # Use mean pooling
            representation = torch.mean(transformer_out, dim=1)
        
        # Generate prediction
        output = self.output_net(representation)
        
        # Return attention weights from last layer for visualization
        # Note: For true attention weights, we'd need to modify TransformerEncoder
        attention_weights = None
        
        return output, attention_weights

class EnhancedTransformerPredictor(nn.Module):
    """Transformer with Physics-Informed features"""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.3):
        super(EnhancedTransformerPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Physics layer
        self.physics_layer = DiurnalAwarePhysicsLayer(input_size, d_model)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, x, physics_features):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x)
        
        # Extract CLS representation
        transformer_repr = transformer_out[:, 0, :]
        
        # Physics processing
        physics_out = self.physics_layer(physics_features)
        
        # Fusion
        combined = torch.cat([transformer_repr, physics_out], dim=1)
        fused = self.fusion(combined)
        
        # Output
        output = self.output_net(fused)
        
        return output, None
