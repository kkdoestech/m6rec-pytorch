# Phase 5: The Two-Tower Model & Training Pipeline

Welcome to Phase 5! We are now connecting all the puzzle pieces from the previous phases into a single, cohesive PyTorch training pipeline.

## The Full Data Flow
In a PyTorch training loop, data flows in a continuous cycle of prediction, error calculation, and learning. Here is the step-by-step journey of our data in the M6-Rec architecture:

1. **Raw Text:** We start with raw user history and candidate item text.
2. **Tokenizer:** The text is formatted with our special `[BOS]` and `[EOS]` markers and converted into numerical `input_ids` and `attention_mask` tensors.
3. **DistilBERT Encoders (The Two Towers):** The numerical tensors are passed into our `M6RecModel`. The model uses a DistilBERT backbone to read the text. It acts as a "Two-Tower" model because we encode the user history and the item text independently to get two separate representations.
4. **[CLS] Embeddings:** We extract the output corresponding to the `[CLS]` token (the very first token) from the DistilBERT outputs. This dense vector represents the semantic meaning of the entire text sequence.
5. **InfoNCE Loss:** We pass the user `[CLS]` embeddings and the item `[CLS]` embeddings into our `InfoNCELoss` function. It calculates a similarity grid (using Dot Product) and computes how wrong the model's predictions are.
6. **Backpropagation (`loss.backward()`):** We compute the gradients—the mathematical "direction" we need to shift our model's weights to make the loss smaller.
7. **Optimizer Step (`optimizer.step()`):** The optimizer (AdamW) takes those gradients and actually updates the DistilBERT weights so it makes better predictions next time!

## Understanding Backpropagation (Intuition for Beginners)
If you've taken a first-year calculus course, you know about **derivatives**. A derivative simply measures the "rate of change"—if I change $x$ a tiny bit, how much does $y$ change?

In deep learning:
- $y$ is our **Loss** (how bad the model is performing).
- $x$ represents the millions of **Weights** (parameters) inside DistilBERT.

**Backpropagation** is just applying the **Chain Rule** from calculus across the entire neural network. When we call `loss.backward()`, PyTorch calculates the partial derivative (gradient) of the Loss with respect to every single weight in the model: $\frac{\partial \text{Loss}}{\partial \text{Weight}}$.

- If the gradient is **positive**, increasing the weight makes the loss go up (which is bad!). So, the optimizer will *decrease* that weight.
- If the gradient is **negative**, increasing the weight makes the loss go down (which is good!). So, the optimizer will *increase* that weight.

### The Push and Pull of Contrastive Learning
In our InfoNCE Loss:
*   We want **User A** to be close to **Item A** (Positive Pair).
*   We want **User A** to be far from **Item B** (Negative Pair).

When backpropagation happens, it updates the weights inside DistilBERT so that the next time it reads User A's history, the output vector is physically pushed closer in the 768-dimensional space to Item A's vector, and pushed away from Item B's vector. The model is literally learning a new geometry for the words!
