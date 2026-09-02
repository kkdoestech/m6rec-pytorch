"""
Phase 5:



"""




import os
import sys
from pathlib import Path

# Ensure the root project directory is in the Python search path for clean imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from transformers import AutoModel

class M6RecModel(nn.Module):
    """
    The Two-Tower M6-Rec Model.
    
    Architecture:
      - Backbone: Pretrained DistilBERT (Base Encoder)
      - Head: Linear Projection Layer (transforms 768-dim [CLS] token to projection_dim)
      
    Memory Optimization:
      - Setting freeze_backbone=True sets `requires_grad = False` on all DistilBERT
        parameters. This prevents PyTorch from storing massive activation graphs
        and calculating 66M+ gradients on low-spec/CPU environments.
    """
    def __init__(self, model_name="distilbert-base-uncased", projection_dim=128, freeze_backbone=True):
        super(M6RecModel, self).__init__()
        
        # 1. Load Pre-trained Transformer Encoder (DistilBERT)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 2. Freeze Transformer Backbone (if requested)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] DistilBERT backbone parameters successfully frozen (requires_grad=False).")
        
        # 3. Trainable Projection Layer (768 -> 128)
        # Even with frozen backbone, this layer trains fast and keeps memory footprint minimal!
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)
        
    def forward(self, user_input_ids, user_attention_mask, item_input_ids, item_attention_mask):
        """
        Forward pass for both User and Item towers.
        
        Args:
            user_input_ids: Tensor of shape (Batch_Size, Seq_Len)
            user_attention_mask: Tensor of shape (Batch_Size, Seq_Len)
            item_input_ids: Tensor of shape (Batch_Size, Seq_Len)
            item_attention_mask: Tensor of shape (Batch_Size, Seq_Len)
            
        Returns:
            user_embeddings: Tensor of shape (Batch_Size, projection_dim) -> e.g., (2, 128)
            item_embeddings: Tensor of shape (Batch_Size, projection_dim) -> e.g., (2, 128)
        """
        # --- User Tower ---
        user_outputs = self.encoder(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask
        )
        user_cls = user_outputs.last_hidden_state[:, 0, :] # Extract [CLS] vector (Batch, 768)
        user_embeddings = self.projection(user_cls)          # Project to (Batch, 128)
        
        # --- Item Tower (Shared Encoder Weights) ---
        item_outputs = self.encoder(
            input_ids=item_input_ids,
            attention_mask=item_attention_mask
        )
        item_cls = item_outputs.last_hidden_state[:, 0, :] # Extract [CLS] vector (Batch, 768)
        item_embeddings = self.projection(item_cls)          # Project to (Batch, 128)
        
        return user_embeddings, item_embeddings

if __name__ == "__main__":
    # Quick sanity check for the model architecture
    print("Testing M6RecModel initialization and dummy forward pass...")
    model = M6RecModel(freeze_backbone=True)
    dummy_ids = torch.ones((2, 32), dtype=torch.long)
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    u_emb, i_emb = model(dummy_ids, dummy_mask, dummy_ids, dummy_mask)
    print(f"User Embedding Shape: {u_emb.shape}")
    print(f"Item Embedding Shape: {i_emb.shape}")
    print("Model Sanity Check Passed!")
