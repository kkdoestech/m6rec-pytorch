# m6rec-pytorch
> A PyTorch-based text-to-text recommender system implementing the M6-Rec foundation model paradigm.

## Overview & Architecture
`m6rec-pytorch` implements the core concepts of the **M6-Rec** paradigm (*Towards Universal-Domain Recommendation by Learning from Web-Scale Pre-trained Multimodal Models*). It eliminates the traditional ID-based collaborative filtering approach which suffers from cold-start issues and closed-domain limitations.

Instead, this project frames user behavior modeling as language modeling. It transforms behavioral clickstreams and item metadata into structured natural language prompts. These prompts are processed by a foundation Transformer model (`distilbert-base-uncased`) to extract rich, dense semantic representations. User-item alignment is then optimized using a contrastive **InfoNCE Loss**.

## Key Features
- **Prompt-Based Data Formatting**: Converts user histories (clicks, views) and candidate items into tokenizable text prompts delimited by semantic boundary markers (`[BOS]`, `[EOS]`).
- **Dense Embedding Extraction**: Extracts contextualized `[CLS]` token representations using the Hugging Face Transformers library.
- **Custom InfoNCE Loss**: PyTorch implementation of in-batch contrastive learning. Utilizes cosine similarity and temperature scaling via Cross-Entropy to pull positive user-item pairs together while pushing negative pairs apart.
- **Zero-Shot & Cold-Start Resilience**: Recommends items based on textual semantics rather than historical ID frequency.

## Project Structure
```text
m6rec-pytorch/
├── src/
│   ├── __init__.py
│   ├── format_data.py           # Prompt construction and Hugging Face tokenization
│   ├── extract_embeddings.py    # Forward passes and [CLS] vector extraction
│   ├── loss.py                  # InfoNCE contrastive loss implementation
│   ├── model.py                 # Two-Tower M6-Rec model architecture
│   ├── test_hf.py               # Hugging Face environment validation
│   └── train.py                 # Training pipeline & checkpoint saver
├── docs/
│   ├── phase4_infonce_loss.md       # InfoNCE mathematical specification
│   └── phase5_training_pipeline.md  # Training loop & backpropagation guide
├── experiments/                 # Saved model weights & checkpoints
├── requirements.txt             # Project dependencies
└── .gitignore                   # Standard Python/PyTorch gitignore
```

## Installation & Setup Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kkdoestech/m6rec-pytorch.git
   cd m6rec-pytorch
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart / Usage Examples

### 1. Formatting and Tokenizing Data
```python
from src.format_data import format_m6_prompt

prompt = format_m6_prompt(
    user_history=["clicked blue Nike running shoes", "viewed adidas sports bottle"],
    candidate_item="Puma lightweight training shorts"
)
print(prompt)
# Output: [BOS] clicked blue Nike running shoes , viewed adidas sports bottle [EOS] [BOS] Puma lightweight training shorts [EOS]
```

### 2. Extracting Embeddings
Run the extraction script to pass a batch of text through the Transformer and retrieve the 2D `[CLS]` embeddings:
```bash
python src/extract_embeddings.py
```

### 3. Calculating InfoNCE Loss
You can test the custom loss function directly to see how `[Batch, Hidden]` vectors are processed into a `[Batch, Batch]` similarity matrix:
```bash
python src/loss.py
```

### 4. Training the Two-Tower Model
Run the complete training pipeline to train the model and save a checkpoint:
```bash
python src/train.py
```
