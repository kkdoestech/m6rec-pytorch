"""

In Phases 1->3, we used a frozen DistilBERT (meaning we didn't change its weights) just to get numbers. 
    -> Now, in Phase 4, we are going to define a loss function that will tell DistilBERT how to improve 
    its numbers so they become useful for recommendation.



High-Level Concept: What is InfoNCE Loss?
    Imagine you're a teacher giving a multiple-choice quiz to 4 students.
        + Each student (User) has a question (their purchase history).
        + Each student also has 4 possible answers (4 items).
        + The correct answer for each student is the item they actually purchased (the positive pair).
            => ex: The correct ans for Student 0 is Item 0, for Student 1 is Item 1, etc.
    The InfoNCE loss turns the embedding-learning problem -> into exactly this multiple-choice quiz.
        + We have a batch of (B) users and (B) items (each user bought exactly one of these items).
        + The Model produces a vector (embedding) for each user (Query) and each item (Item).
        + We compute a similarity score between every user and every item (a (B x B) grid).
        + We tell PyTorch: "The correct answer for row (i) -> is column i (the diagonal)."
        + If the model gives a high score to the diagonal (the correct item) and low scores to
        everything else -> the loss is small
            -> if it messes up -> the loss is huge, and we update the model to fix it.



Breaking Down (loss.py) -> Line by Line:
    1. The Import Section:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

            + (torch): PyTorch library -> our calculator
            + (torch.nn): Contains pre-built neural network layers (like (Linear), (CrossEntropyLoss)).
                -> we alias it as nn.
            + (torch.nn.functional): Contains mathematical functions that don't have internal
            parameters (like (normalize), (softmax), (relu)). => we alias it as F.

    
    2. The Class Definition:
        class InfoNCELoss(nn.Module):
        def __init__(self, temperature=0.07):
            super(InfoNCELoss, self).__init__()
            self.temperature = temperature
            self.cross_entropy = nn.CrossEntropyLoss()

                + class InfoNCELoss(nn.Module) -> we're creating a new class that inherits from
                PyTorch's nn.Module -> this is the standard way to build custom layers in PyTorch.
                
                + __init__ -> the constructor

                    + temperature=0.07 -> a hyperparameter that controls how "confident" the model is.
                      + Why 0.07 -> it's a common value -> it will scale the similarity scores.
                        -> A small temperature (like 0.07) makes the model more confident, while a 
                        larger temperature (like 1.0) makes it more "uncertain".
                            -> it exaggerates differences, which helps the model learn faster.

                    + super(InfoNCELoss, self).__init__()
                        -> this calls the constructor of parent class (nn.Module)
                            -> so everything initializes properly.
                    
                    + self.cross_entropy = nn.CrossEntropyLoss()
                        -> we store a pre-built cross-entropy loss function in our class,
                        which we'll use later to compute the final loss.

                        
    3. The Forward Method (where the magic happens):
        def forward(self, queries, items):
        
            + (queries): A tensor of shape (batch_size, hidden_dim) -> eg: (2, 768). These are the [CLS]
            embeddings from your user histories.
            + (items): A tensor of the same shape. -> These are [CLS] embeddings from the candidate items.
            + Important assumption: The items are in the same order as the queries.
                -> So queries[0] -> belongs with items[0] (the positive pair), queries[1] with items[1],
                etc.

        Step 1: Normalize the Vectors
            queries = F.normalize(queries, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
                
                What is normalization?
                    + A vector has a length (magnitude) and a direction (where it points).
                    + Normalization -> makes the length exactly 1, while keeping the direction the same.
                        -> This is called L2 normalization (Euclidean norm).

                Why do we do this?
                    + After normalization, the dot product between 2 vectors -> becomes cosine similarity,
                    which range from -1 -> 1.
                    + This forces the model -> to focus only on the angle (direction) between vectors, not
                    their length.
                        -> In recommendation, we care about similarity of meaning, not the magnitude of
                        the embedding.

                Parameters:
                    + (p=2) -> means L2 norm (standard Euclidean length).
                    + (dim=1) -> means we compute the length across the hidden dimension (the 768 numbers)
                    for each sample independently.

                Example: (tiny 2D vector):
                    + Before: queries[0] = [3.0, 4.0] (length = 5)
                    + After: queries[0] = [3/5, 4/5] = [0.6, 0.8] (length = 1)

        Step 2: Matrix Multplication (The Similarity Grid)
            logits = torch.matmul(queries, items.T)

                + items.T -> the transpose of the items matrix.
                    if (items) has shape (batch_size, hidden_dim) -> then items.T has shape
                    (hidden_dim, batch_size).
                + torch.matmul -> matrix multiplication
                    Shape of queries: (B, H)
                    Shape of items.T: (H, B)
                    Resulting shape: (B, B) -> a square grid.
                + what does this grid contains?
                    + cell (i, j) = dot product of queries[i] and items[j]
                    + because we normalized -> this is exactly the cosine similarity between user i and
                    item j.
                    + Diagonal cells (i, j) = similarity between user (i) and the item they actually 
                    bought (positive pair). -> we want these to be high (close to 1).
                    + Off-diagonal cells (i, j) -> where i != j => similarity between user (i) and items
                    other users bought (negative pairs).
                        -> we want these to be low (close to 0 or negative).

        Step 3: Temperature Scaling
            logits = logits / self.temperature

                + We divide every number in the (B, B) grid by (temperature) (eg: 0.07)
                + Why?
                    + If similarity scores are between -1 and 1 -> they're too small for the cross-entropy
                    loss -> to "care" about small differences.
                    + By dividing by a small number -> we stretch the numbers.
                        -> For ex: 0.8 / 0.07 = around 11.4 -> and 0.2 / 0.07 = around 2.8
                            -> This makes the correct answer stand out much more clearly
                                -> and the loss function will penalize mistakes much harder.
                                    -> it's like turning up the contrast on an image.

        Step 4: Create the Labels (The Answer Key)
            batch_size = queries.shape[0]
            labels = torch.arange(batch_size, device=queries.device)

                + queries.shape[0] -> get the batch size (eg: 2)
                + torch.arange(batch_size) -> create a tensor: [0, 1, 2,..., B-1]
                    + For (B=2) -> labels = [0, 1]
                + device=queries.device -> ensures this tensor -> lives on the same hardware (CPU or
                GPU) -> as the queries.

                + What do these labels mean?
                    + For row i (user i) -> the correct column is labels[i].
                    + So for user 0 -> the correct answer is column 0
                        For user 1 -> the correct answer is column 1
                    + This matches our assumption that items are ordered -> to match the users.

        Step 5: Compute Cross-Entropy Loss
            loss = self.cross_entropy(logits, labels)
            return loss

                + self.cross_entropy -> the PyTorch cross-entropy function -> we stored in __init__
                + What does it do?
                    + It takes the (B, B) grid of scores (logits) and the (B) labels.
                    + For each row (i):
                        + It applies (softmax) to the row: turns the scores into probabilities that sum
                        to 1.
                        + it checks the probability assigned to the correct column labels[i].
                        + The loss is -log(probability_of_correct_class)
                    + If the model gives a high probability to the correct column -> -log(high) is a 
                    small number.
                        -> if it gives a low probability -> -log(low) is a huge number.


    A Tiny Numerical Example (Batch size = 2, hidden_dim = 3)
        Let's replace 768 with 3 -> so we can do the math by hand.
        
        Step 1: Fake inputs (before normalization)
            queries = [[2.0, 1.0, 0.0],    # User 1
                       [0.0, 1.0, 2.0]]    # User 2
            items   = [[1.0, 0.0, 0.0],    # Item 1 (bought by User 1)
                       [0.0, 1.0, 1.0]]    # Item 2 (bought by User 2)

        Step 2: Normalize (Make each vector length = 1)
            + Length of queries[0] = sqrt(2^2 + 1^2 + 0^2) = sqrt(5) = 2.236
                -> queries[0] -> [2 / 2.236, 1 / 2.236, 0] = [0.894, 0.447, 0.000]
            + Length of queries[1] = sqrt(0^2 + 1^2 + 2^2) = sqrt(5) = 2.236
                -> queries[1] = [0.000, 0.447, 0.894]
            + Length of items[0] = sqrt(1^2 + 0^2 + 0^2) = 1
                -> items[0] = [1.000, 0.000, 0.000]
            + Length of items[1] = sqrt(0^2 + 1^2 + 1^2) = sqrt(2) = 1.414
                -> items[1] = [0.000, 0.707, 0.707]
        
        Step 3: Matrix Multiplication (logtis before temperature)
            We compute queries @ items.T:
                Row 0, Col 0: [0.894, 0.447, 0.000] . [1.000, 0.000, 0.000] = 0.894*1 + 0.447*0 + 0*0 
                                                                            = 0.894
                
                Row 0, Col 1: [0.894, 0.447, 0.000] · [0.000, 0.707, 0.707] = 0.894*0 + 0.447*0.707 
                                                                            + 0*0.707 = 0.316
                        
                Row 1, Col 0: [0.000, 0.447, 0.894] · [1.000, 0.000, 0.000] = 0.000
                
                Row 1, Col 1: [0.000, 0.447, 0.894] · [0.000, 0.707, 0.707] = 0.447*0.707 + 0.894*0.707 
                                                                            = 0.316 + 0.632 = 0.948
            
            Our logits grid (similarities) before temperature:
                [[0.894, 0.316],
                 [0.000, 0.948]]

                    + Diagonal: User1-Item1 = 0.894 (good), User2-Item2 = 0.948 (good).
                    + Off-diagonal: User1-Item2 = 0.316 (not zero but lower), User2-Item1 = 0.000(perfect)

                        We have 2 users and 2 items:
                            + User1 bought Item1 (Positive pair)
                            + User2 bought Item2 (Positive pair)
                        when we look at our grid -> we have 4 pairs total:
                            Pair	        Type	                    What we WANT the score to be
                            User 1-Item 1	Positive (Correct answer)	HIGH (close to 1)
                            User 2-Item 2	Positive (Correct answer)	HIGH (close to 1)
                            User 1-Item 2	Negative (Wrong answer)	    LOW (close to 0)
                            User 2-Item 1	Negative (Wrong answer)	    LOW (close to 0)

                The pair User2 - Item1 has a score of 0.000.
                Let's think about what this number actually means in math:

                    After normalization, these are cosine similarities.

                    A cosine similarity of 1.0 means the two vectors point in the exact same 
                    direction (identical meaning).

                    A cosine similarity of 0.0 means the two vectors are perfectly perpendicular
                    (at a 90-degree angle). They have absolutely nothing in common.

                    Now, ask yourself: Should User 2 have anything in common with Item 1?
                    No! Because Item 1 belongs to User 1. We want User 2 to completely ignore 
                    Item 1.

                    So, when the model gives a score of 0.000 for this negative pair, it is 
                    saying:
                        "I have no idea why you would show Item 1 to User 2. 
                        They are completely unrelated."


                Now look at the other negative pair: User1 - Item2 scored 0.316.

                0.316 is higher than 0.000.

                This means the model is a little confused. It thinks User 1 and Item 2 share some 
                slight similarity (31.6% similar), even though they shouldn't.

                If the model had to guess, it might accidentally pick Item 2 for User 1 because 
                0.316 is not zero.

                So, between the two negative pairs:
                    0.316 = Bad (the model is a bit confused).

                    0.000 = Perfect (the model is absolutely certain this is the wrong item).

    In InfoNCE loss:
        + For the Diagonal (Positive pairs): HIGHER = BETTER. You want them running towards 1.0.
        + For the Off-Diagonal (Negative pairs): LOWER = BETTER. You want them running towards 0.0 
        (or even negative numbers).


        Step 4: Apply Temperature (let's use 0.1)
            Divide every number by 0.1 (multiply by 10)
                [[8.94, 3.16],
                 [0.00, 9.48]]

        Step 5: Labels
            labels = [0, 1] -> meaning row 0 correct column is 0, row 1 correct column is 1.

        Step 6: Cross-Entropy Loss
            For Row 0: scores = [8.94, 3.16]
            + Softmax:
                + exp(8.94) ≈ 7610, exp(3.16) ≈ 23.6
                + Prob(col 0) = 7610 / (7610+23.6) ≈ 0.997
                + Prob(col 1) ≈ 0.003
            + Loss for row 0 = -log(0.997) ≈ 0.003 (tiny penalty, model is confident and correct).

            For Row 1: scores = [0.00, 9.48]
            + Softmax:
                + exp(0) = 1, exp(9.48) ≈ 13000
                + Prob(col 0) ≈ 1 / 13001 ≈ 0.00008
                + Prob(col 1) ≈ 13000 / 13001 ≈ 0.99992
            + Loss for row 1 = -log(0.99992) ≈ 0.00008 (even tinier penalty).

            Total loss = average of row losses = (0.003 + 0.00008) / 2 ≈ 0.0015.
                -> This is a very low loss because the model already understood the 
                relationships perfectly.

                



"""

'''
4. The Final Loss Value
    Final Calculated Loss: 0.6183

    What does this number mean? It is a positive floating-point number representing the 
    average penalty (error) the model received on this tiny fake batch.

    Is 0.6183 a good or bad number?
        Because your fake_queries and fake_items are completely random 
        (they have no real relationship), the model has no idea which item belongs to which user.

        In a 2-class multiple-choice quiz, if the model guesses randomly, the average loss 
        is around log(2) ≈ 0.693.

        You got 0.6183, which is very close to 0.693! The slight difference is just because 
        of the random numbers.

        If the loss was 0.0, that would mean the model is perfectly confident 
        (which would be suspicious for random data). If it was 5.0, something would be broken.

'''












'''

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


'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive Loss) for training representations:
        This pulls positive pairs together -> and pushes in-batch negative pairs apart.
    """
    def __init__(self, temperature=0.07):
        # we inherit from nn.Module -> to make this a professional, reusable PyTorch layer.
        super(InfoNCELoss, self).__init__()

        # temperature is a hyperparameter that controls the "sharpness" of the logits.
        # smaller temperature (eg: 0.07) -> makes the model more confident/punishes errors harder.
        self.temperature = temperature

        # we will use CrossEntropyLoss, treating the contrastive task as a classification task.
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, queries, items):
        """
        Calculate the InfoNCE Loss between a batch of queries and a batch of items.

        Args:
            queries: Tensor of shape (Batch_Size, Hidden_Dim) -> eg: (2, 768)
            items:  Tensor of shape (Batch_Size, Hidden_Dim) -> eg: (2, 768)

        Returns:
            loss: A scalar PyTorch tensor -> containing the loss value.
        """
        # 1. Normalize the vectors -> so their length is 1
        # This turns the Dot Product into a mathematically pure "Cosine Similarity".
        # It forces the model -> to focus on the DIRECTION of the vector, not the magnitude.
        queries = F.normalize(queries, p=2, dim=1)
        items = F.normalize(items, p=2, dim=1)

        print(f"DEBUG: Queries shape normalized: {queries.shape}")
        print(f"DEBUG: Items shape normalized: {items.shape}")

        # 2. Matrix Multiplication (Dot Product across the whole batch at once)
        # We do: [Batch, 768] @ [768, Batch] -> [Batch, Batch]
        # In python -> .T -> transposes the matrix (flip dimensions)
        logits = torch.matmul(queries, items.T)

        print(f"DEBUG: Logits (Similarity Grid) shape: {logits.shape}")

        # 3. Apply Temperature Scaling
        # We divide the similarity scores -> by the temperature
        logits = logits / self.temperature

        # 4. Create the labels (The "Answers" for our multiple-choice test)
        # Our positive pairs -> are always exactly on the diagonal.
        # So for Row 0 -> the correct column is 0 -> for Row 1 -> the correct column is 1.
        # torch.arange(Batch) -> generates a list like [0, 1, 2,..., Batch - 1]
        batch_size = queries.shape[0]
        labels = torch.arange(batch_size, device=queries.device)

        print(f"DEBUG: Labels shape: {labels.shape}")
        print(f"DEBUG: Labels values: {labels}")

        # 5. Calculate loss
        # We feed our [Batch, Batch] logits grid and our [Batch] labels into CrossEntropy
        loss = self.cross_entropy(logits, labels)
        
        return loss

# Let's run a quick numerical test 
if __name__ == "__main__":
    # create a fake embedding vectors to simulate our DistilBERT [CLS] output
    # batch size = 2, hidden dim = 768
    fake_queries = torch.rand(2, 768)
    fake_items = torch.rand(2, 768)

    # initialize our custom loss function
    loss_fn = InfoNCELoss(temperature=0.07)

    # Calculate loss
    print("Starting Forward Pass...\n")
    loss_val = loss_fn(fake_queries, fake_items)

    print(f"\nFinal Calculated Loss: {loss_val.item():.4f}")
