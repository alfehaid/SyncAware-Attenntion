"""
🧠 DT-HybridNet: Digital Twin Hybrid Deep Learning Model
The world's first CNN-LSTM-Transformer fusion model for Digital Twin IDS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HybridNetConfig:
    """Configuration for DT-HybridNet model"""
    
    # Input dimensions
    network_features: int = 41      # Traditional network features
    dt_features: int = 20          # Digital Twin features
    system_features: int = 15      # System metrics
    
    # CNN Branch
    cnn_filters: List[int] = (64, 128, 256)
    cnn_kernel_sizes: List[int] = (3, 3, 3)
    cnn_dropout: float = 0.2
    
    # LSTM Branch
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True
    
    # Transformer Fusion
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dim: int = 256
    transformer_dropout: float = 0.1
    
    # Fusion Strategy
    fusion_method: str = "adaptive"  # "concat", "attention", "adaptive"
    fusion_dim: int = 512
    
    # Output
    num_classes: int = 2  # Binary classification
    num_attack_types: int = 8  # For multi-task learning
    
    # Training
    dropout_rate: float = 0.3
    activation: str = "gelu"

class SyncAwareAttention(nn.Module):
    """
    Novel Sync-Aware Attention Mechanism for Digital Twin synchronization patterns
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Standard attention components
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Sync-aware gating mechanism
        self.sync_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_heads),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, sync_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sync-aware gating if sync features provided
        if sync_features is not None:
            sync_weights = self.sync_gate(sync_features)  # [batch_size, num_heads]
            sync_weights = sync_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_heads, 1, 1]
            scores = scores * sync_weights
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Average attention across heads

class CNNBranch(nn.Module):
    """
    CNN Branch for network traffic pattern recognition
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        # 1D CNN layers for network features
        layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(config.cnn_filters, config.cnn_kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config.cnn_dropout)
            ])
            in_channels = out_channels
            
            # Add attention after each conv block
            if i > 0:
                layers.append(ChannelAttention(out_channels))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(config.cnn_filters[-1], config.fusion_dim // 3),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, features]
        batch_size = x.size(0)
        
        # Reshape for 1D CNN: [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Global pooling: [batch_size, channels, 1] -> [batch_size, channels]
        x = self.global_pool(x).squeeze(-1)
        
        # Project to fusion dimension
        x = self.projection(x)
        
        return x

class ChannelAttention(nn.Module):
    """Channel attention mechanism for CNN"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.size()
        
        # Global pooling
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)
        
        # Attention weights
        avg_weights = self.fc(avg_pool)
        max_weights = self.fc(max_pool)
        weights = (avg_weights + max_weights).view(batch_size, channels, 1)
        
        return x * weights

class LSTMBranch(nn.Module):
    """
    LSTM Branch for temporal dynamics in Digital Twin features
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.dt_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional,
            batch_first=True
        )
        
        # Temporal attention
        lstm_output_size = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_size, config.fusion_dim // 3),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, dt_features]
        batch_size = x.size(0)
        
        # Create sequence from features (simulate temporal data)
        seq_len = 10  # Simulate 10-step sequence
        x_seq = x.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, dt_features]
        
        # Add noise to simulate temporal variations
        noise = torch.randn_like(x_seq) * 0.01
        x_seq = x_seq + noise
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_seq)
        
        # Apply temporal attention
        attn_weights = self.temporal_attention(lstm_out)  # [batch_size, seq_len, 1]
        
        # Weighted sum of LSTM outputs
        attended_output = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, lstm_output_size]
        
        # Project to fusion dimension
        output = self.projection(attended_output)
        
        return output

class DenseBranch(nn.Module):
    """
    Dense Branch for system metrics and other features
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        self.layers = nn.Sequential(
            nn.Linear(config.system_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(64, config.fusion_dim // 3),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TransformerFusion(nn.Module):
    """
    Transformer-based fusion layer for combining all branches
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        # Input projection to transformer dimension
        branch_dim = config.fusion_dim // 3
        self.input_projection = nn.Linear(branch_dim, config.transformer_dim)
        
        # Positional encoding for branches
        self.branch_embeddings = nn.Parameter(torch.randn(3, config.transformer_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.transformer_dropout,
            activation=config.activation,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        # Sync-aware attention
        self.sync_attention = SyncAwareAttention(config.transformer_dim, config.transformer_heads)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.transformer_dim * 3, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, cnn_out: torch.Tensor, lstm_out: torch.Tensor, dense_out: torch.Tensor, 
                sync_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = cnn_out.size(0)
        
        # Stack branch outputs
        branch_outputs = torch.stack([cnn_out, lstm_out, dense_out], dim=1)  # [batch_size, 3, branch_dim]
        
        # Project to transformer dimension
        x = self.input_projection(branch_outputs)  # [batch_size, 3, transformer_dim]
        
        # Add branch embeddings
        x = x + self.branch_embeddings.unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply sync-aware attention if sync features available
        if sync_features is not None:
            x, attn_weights = self.sync_attention(x, sync_features)
        
        # Flatten and project
        x = x.view(batch_size, -1)  # [batch_size, 3 * transformer_dim]
        output = self.output_projection(x)
        
        return output

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion strategy with dynamic weighting
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        branch_dim = config.fusion_dim // 3
        
        # Attack type predictor for adaptive weighting
        self.attack_predictor = nn.Sequential(
            nn.Linear(branch_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_attack_types),
            nn.Softmax(dim=-1)
        )
        
        # Fusion weight generators
        self.weight_generators = nn.ModuleDict({
            'cnn': nn.Sequential(
                nn.Linear(config.num_attack_types + branch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'lstm': nn.Sequential(
                nn.Linear(config.num_attack_types + branch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'dense': nn.Sequential(
                nn.Linear(config.num_attack_types + branch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        })
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(branch_dim * 3, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, cnn_out: torch.Tensor, lstm_out: torch.Tensor, dense_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate all features
        all_features = torch.cat([cnn_out, lstm_out, dense_out], dim=-1)
        
        # Predict attack type probabilities
        attack_probs = self.attack_predictor(all_features)
        
        # Generate adaptive weights
        cnn_weight = self.weight_generators['cnn'](torch.cat([attack_probs, cnn_out], dim=-1))
        lstm_weight = self.weight_generators['lstm'](torch.cat([attack_probs, lstm_out], dim=-1))
        dense_weight = self.weight_generators['dense'](torch.cat([attack_probs, dense_out], dim=-1))
        
        # Apply weights
        weighted_features = torch.cat([
            cnn_out * cnn_weight,
            lstm_out * lstm_weight,
            dense_out * dense_weight
        ], dim=-1)
        
        # Final fusion
        fused_output = self.fusion_layer(weighted_features)
        
        return fused_output, attack_probs

class DTHybridNet(nn.Module):
    """
    DT-HybridNet: Digital Twin Hybrid Deep Learning Model
    CNN + LSTM + Transformer + Adaptive Fusion for Digital Twin IDS
    """
    
    def __init__(self, config: HybridNetConfig):
        super().__init__()
        self.config = config
        
        logger.info("🧠 Initializing DT-HybridNet...")
        
        # Feature branches
        self.cnn_branch = CNNBranch(config)
        self.lstm_branch = LSTMBranch(config)
        self.dense_branch = DenseBranch(config)
        
        # Fusion strategies
        if config.fusion_method == "transformer":
            self.fusion = TransformerFusion(config)
        elif config.fusion_method == "adaptive":
            self.fusion = AdaptiveFusion(config)
        else:
            # Simple concatenation
            self.fusion = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        # Classification heads
        self.binary_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_classes)
        )
        
        self.multiclass_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_attack_types)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info("✅ DT-HybridNet initialized successfully!")
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        total_features = x.size(1)
        
        # Split features into branches
        network_end = self.config.network_features
        dt_end = network_end + self.config.dt_features
        
        network_features = x[:, :network_end]
        dt_features = x[:, network_end:dt_end]
        system_features = x[:, dt_end:]
        
        # Forward through branches
        cnn_out = self.cnn_branch(network_features)
        lstm_out = self.lstm_branch(dt_features)
        dense_out = self.dense_branch(system_features)
        
        # Fusion
        if self.config.fusion_method == "adaptive":
            fused_features, attack_type_probs = self.fusion(cnn_out, lstm_out, dense_out)
        elif self.config.fusion_method == "transformer":
            # Extract sync features for transformer
            sync_features = dt_features[:, :5] if dt_features.size(1) >= 5 else None
            fused_features = self.fusion(cnn_out, lstm_out, dense_out, sync_features)
            attack_type_probs = None
        else:
            # Simple concatenation
            fused_features = torch.cat([cnn_out, lstm_out, dense_out], dim=-1)
            fused_features = self.fusion(fused_features)
            attack_type_probs = None
        
        # Classification
        binary_logits = self.binary_classifier(fused_features)
        multiclass_logits = self.multiclass_classifier(fused_features)
        
        # Return outputs
        outputs = {
            'binary_logits': binary_logits,
            'multiclass_logits': multiclass_logits,
            'fused_features': fused_features,
            'branch_outputs': {
                'cnn': cnn_out,
                'lstm': lstm_out,
                'dense': dense_out
            }
        }
        
        if attack_type_probs is not None:
            outputs['attack_type_probs'] = attack_type_probs
        
        return outputs
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'DT-HybridNet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config.__dict__,
            'branches': ['CNN', 'LSTM', 'Dense'],
            'fusion_method': self.config.fusion_method,
            'novel_components': ['SyncAwareAttention', 'AdaptiveFusion', 'ChannelAttention']
        }

def create_dt_hybrid_net(
    input_features: int = 76,
    fusion_method: str = "adaptive",
    num_classes: int = 2
) -> DTHybridNet:
    """Create DT-HybridNet model with specified configuration"""
    
    config = HybridNetConfig(
        network_features=41,
        dt_features=20,
        system_features=input_features - 61,  # Remaining features
        fusion_method=fusion_method,
        num_classes=num_classes
    )
    
    model = DTHybridNet(config)
    
    logger.info(f"📊 Created DT-HybridNet:")
    model_info = model.get_model_info()
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Fusion method: {model_info['fusion_method']}")
    logger.info(f"  Novel components: {model_info['novel_components']}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    logger.info("🧪 Testing DT-HybridNet creation...")
    
    model = create_dt_hybrid_net(input_features=56, fusion_method="adaptive")
    
    # Test forward pass
    batch_size = 32
    input_features = 56
    test_input = torch.randn(batch_size, input_features)
    
    with torch.no_grad():
        outputs = model(test_input)
    
    logger.info("✅ DT-HybridNet test completed successfully!")
    logger.info(f"Binary output shape: {outputs['binary_logits'].shape}")
    logger.info(f"Multiclass output shape: {outputs['multiclass_logits'].shape}")
    logger.info(f"Fused features shape: {outputs['fused_features'].shape}")