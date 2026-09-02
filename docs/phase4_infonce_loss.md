# Phase 4: InfoNCE Contrastive Loss

Welcome to Phase 4! In this phase, we move beyond simply extracting vector embeddings and start **training** our models to understand relationships. 

## The Core Concept: Magnetic Vectors
In recommendation systems, we want a user's prompt (the **Query**) and the ideal product they will buy (the **Item**) to be mathematically identical, or at least very close. 
Think of these vectors as magnets. We want to:
1. **Pull** the User Query and their *actual* purchased item closer together in space (Positive Pair).
2. **Push** the User Query away from *random* items they didn't buy (Negative Pairs).

## The Math: Dot Products as Similarity (`x @ y.T`)
How do we measure if two vectors are close? We use the **Dot Product**.
If vector `A` points in the exact same direction as vector `B`, their dot product is a large positive number. If they point in opposite directions, it's a negative number. If they are perpendicular (unrelated), it's close to zero.

### The Matrix Multiplication Grid
Imagine you have a batch of 4 Users (`Batch = 4`) and the 4 Items they bought.
Both the Queries (`Q`) and Items (`I`) have a shape of `(4, 768)`.

To compare *every* User against *every* Item simultaneously, we use Matrix Multiplication (Linear Algebra):
`Scores = Q @ I.T`

Here, `I.T` means we **transpose** the Item matrix, turning its shape from `(4, 768)` into `(768, 4)`.

When we multiply `(4, 768)` by `(768, 4)`, the inner `768` dimensions collapse, yielding a `(4, 4)` matrix:

```text
       I_0    I_1    I_2    I_3
    +-------------------------
Q_0 | Q0*I0  Q0*I1  Q0*I2  Q0*I3
Q_1 | Q1*I0  Q1*I1  Q1*I2  Q1*I3
Q_2 | Q2*I0  Q2*I1  Q2*I2  Q2*I3
Q_3 | Q3*I0  Q3*I1  Q3*I2  Q3*I3
```

This forms a grid (a 2D tensor):
- The **diagonal** (`Q0*I0`, `Q1*I1`, etc.) represents the **Positive Pairs** (the item the user actually bought). We want these scores to be very HIGH.
- The **off-diagonals** (`Q0*I1`, `Q2*I0`, etc.) represent the **Negative Pairs** (items other users bought). We want these scores to be very LOW.

## InfoNCE: Treating Similarity as a Multiple Choice Quiz
The beauty of the **InfoNCE** (Information Noise-Contrastive Estimation) loss function is that it treats this exact `(4, 4)` grid as a multiple-choice test.

For User 0 (Row 0), there are 4 choices (columns). The "correct" answer is column 0.
For User 1 (Row 1), there are 4 choices (columns). The "correct" answer is column 1.

We can simply feed this grid into standard `CrossEntropyLoss` (the same math used to classify images of cats vs. dogs) and tell PyTorch: *"The correct answers are always along the diagonal!"*

### Temperature Scaling
Before passing the scores to the loss function, we usually divide them by a small number called `temperature` (e.g., 0.07). This mathematically sharpens the differences between scores, making the model penalize itself much harder when it gets the answer wrong.




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
