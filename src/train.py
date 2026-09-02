import os
import sys
from pathlib import Path

# Ensure root directory is accessible for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from transformers import AutoTokenizer

# Import custom M6-Rec modules
from src.model import M6RecModel
from src.loss import InfoNCELoss

def main():
    print("=" * 60)
    print("=== M6-Rec Phase 5: Training Pipeline (CPU Optimized) ===")
    print("=" * 60)
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/6] Running on device: {device}")
    
    # 2. Tokenizer & Model Instantiation
    print("\n[2/6] Instantiating Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Instantiate model with frozen backbone for CPU memory efficiency
    model = M6RecModel(model_name="distilbert-base-uncased", projection_dim=128, freeze_backbone=True).to(device)
    loss_fn = InfoNCELoss(temperature=0.07).to(device)
    print("[INFO] Model and Loss function successfully initialized!")
    
    # 3. Optimizer Setup (Optimizing only trainable projection parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"[3/6] Setting up AdamW optimizer (Trainable parameters: {total_trainable:,})...")
    optimizer = optim.AdamW(trainable_params, lr=1e-3)
    
    # 4. Prepare Sample Batch Data (Batch Size = 2, Max Length = 32)
    print("\n[4/6] Preparing and tokenizing synthetic batch data (Batch Size = 2, Max Length = 32)...")
    raw_user_prompts = [
        "[BOS] clicked blue Nike running shoes , viewed adidas sports bottle [EOS]",
        "[BOS] bought mechanical keyboard , viewed gaming mouse pad [EOS]"
    ]
    raw_item_prompts = [
        "[BOS] Puma lightweight training shorts [EOS]",
        "[BOS] wireless ergonomic mouse [EOS]"
    ]
    
    user_encoded = tokenizer(
        raw_user_prompts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    ).to(device)
    
    item_encoded = tokenizer(
        raw_item_prompts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    ).to(device)
    
    print(f" -> User input_ids shape: {user_encoded['input_ids'].shape}")
    print(f" -> Item input_ids shape: {item_encoded['input_ids'].shape}")
    
    # 5. Training Loop (2 Steps for verification and rapid feedback)
    print("\n[5/6] Starting Training Steps...")
    model.train()
    
    num_steps = 3
    for step in range(num_steps):
        print(f"\n--- [Step {step + 1}/{num_steps}] ---")
        
        # A. Zero Gradients
        print(" -> [A] Resetting gradients (optimizer.zero_grad())...")
        optimizer.zero_grad()
        
        # B. Forward Pass
        print(" -> [B] Executing forward pass through Two-Tower model...")
        user_embeddings, item_embeddings = model(
            user_input_ids=user_encoded['input_ids'],
            user_attention_mask=user_encoded['attention_mask'],
            item_input_ids=item_encoded['input_ids'],
            item_attention_mask=item_encoded['attention_mask']
        )
        print(f"    User embeddings shape: {user_embeddings.shape}")
        print(f"    Item embeddings shape: {item_embeddings.shape}")
        
        # C. Calculate Loss
        print(" -> [C] Computing InfoNCE contrastive loss...")
        loss = loss_fn(user_embeddings, item_embeddings)
        print(f"    Current Loss Value: {loss.item():.4f}")
        
        # D. Backpropagation
        print(" -> [D] Calculating gradients via backpropagation (loss.backward())...")
        loss.backward()
        print("    Gradients computed successfully!")
        
        # E. Optimizer Step
        print(" -> [E] Updating projection weights (optimizer.step())...")
        optimizer.step()
        print(f" -> Step {step + 1} completed! Loss: {loss.item():.4f}")
        
    # 6. Save Model Checkpoint
    print("\n[6/6] Saving trained model checkpoint...")
    os.makedirs("experiments", exist_ok=True)
    save_path = "experiments/m6rec_phase5.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint successfully saved to: {save_path}")
    print("\n" + "=" * 60)
    print("=== Training Pipeline Completed Successfully! ===")
    print("=" * 60)

if __name__ == "__main__":
    main()
