"""
High-Level Concept: What is InfoNCE Loss?
    Imagine you're a teacher giving a multiple-choice quiz to 4 students.
        + Each student (User) has a question (their purchase )


"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive Loss) for training representations.
    This pulls positive pairs together and pushes in-batch negative pairs apart.
    """
    def __init__(self, temperature=0.07):
        # We inherit from nn.Module to make this a professional, reusable PyTorch layer.
        super(InfoNCELoss, self).__init__()
        
        # Temperature is a hyperparameter that controls the "sharpness" of the logits.
        # Smaller temperature (e.g., 0.07) makes the model more confident/punishes errors harder.
        self.temperature = temperature
        
        # We will use CrossEntropyLoss, treating the contrastive task as a classification task.
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, queries, items):
        """
        Calculates the InfoNCE Loss between a batch of queries and a batch of items.
        
        Args:
            queries: Tensor of shape (Batch_Size, Hidden_Dim) -> e.g., (2, 768)
            items:   Tensor of shape (Batch_Size, Hidden_Dim) -> e.g., (2, 768)
            
        Returns:
            loss: A scalar PyTorch tensor containing the loss value.
        """
        # 1. Normalize the vectors so their length is 1. 
        # This turns the Dot Product into a mathematically pure "Cosine Similarity".
        # It forces the model to focus on the DIRECTION of the vector, not the magnitude.
        queries = F.normalize(queries, p=2, dim=1)
        items = F.normalize(items, p=2, dim=1)
        
        print(f"DEBUG: Queries shape normalized: {queries.shape}")
        print(f"DEBUG: Items shape normalized: {items.shape}")

        # 2. Matrix Multiplication (Dot Product across the whole batch at once)
        # We do: [Batch, 768] @ [768, Batch] -> [Batch, Batch]
        # In Python, .T transposes the matrix (flips dimensions).
        logits = torch.matmul(queries, items.T)
        
        print(f"DEBUG: Logits (Similarity Grid) shape: {logits.shape}")

        # 3. Apply Temperature Scaling
        # We divide the similarity scores by the temperature.
        logits = logits / self.temperature

        # 4. Create the Labels (The "Answers" for our multiple-choice test)
        # Our positive pairs are always exactly on the diagonal.
        # So for Row 0, the correct column is 0. For Row 1, the correct column is 1.
        # torch.arange(Batch) generates a list like: [0, 1, 2, ..., Batch-1]
        batch_size = queries.shape[0]
        labels = torch.arange(batch_size, device=queries.device)
        
        print(f"DEBUG: Labels shape: {labels.shape}")
        print(f"DEBUG: Labels values: {labels}")

        # 5. Calculate Loss
        # We feed our [Batch, Batch] logits grid and our [Batch] labels into CrossEntropy.
        loss = self.cross_entropy(logits, labels)
        
        return loss

# --- Let's run a quick numerical test if this file is executed directly ---
if __name__ == "__main__":
    # Create fake embedding vectors to simulate our DistilBERT [CLS] outputs
    # Batch Size = 2, Hidden Dim = 768
    fake_queries = torch.rand(2, 768)
    fake_items = torch.rand(2, 768)
    
    # Initialize our custom loss function
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Calculate loss
    print("Starting Forward Pass...\n")
    loss_val = loss_fn(fake_queries, fake_items)
    
    print(f"\nFinal Calculated Loss: {loss_val.item():.4f}")

