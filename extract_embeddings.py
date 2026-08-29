"""
we have prepared our text and tokenized it -> NLP (format_data.py)
  -> now we do the mathematical heavy lifting: turning text into dense
  semantic vectors.
    -> let's break this down.

Phase 3: Extracting Vector Enbeddings with DistilBERT

  1. Conceptual Foundation: The [CLS] Token and Hidden State
    before we touch the code -> let's understand why we're doing this?
      + What is a Hidden State? -> when a token (like "shoes") passes through the 6 or 12 layers of a Transformer model -> it gets enriched
      with context from surrounding words.
        -> The final output of this journey for a single word -> is called "hidden state" - a massive list of floating-point numbers (
        specifically 768 numbers for DistilBERT).

      + The Problem: -> we don't just want the meaning of the word "shoes". -> we need a single mathematical representation for the ENTIRE
      interaction (User History + Candidate Item) -> to feed into our Recommender System.

      + The [CLS] token: Hugging Face tokenizer -> automatically inject a special token -> called [CLS] (Classify) -> At the very beginning
      of every sequence.
        -> because of the way transformers use "Self-Attention" -> this [CLS] token looks at every other word in the sentence as it passes
        through the layers.
          -> by the time it reaches the final layer -> its hidden state has aggregated the meaning of the entire prompt.

  => Our goal in phase 3: Feed the text in, grab the final layer's output -> and slice out just that [CLS] token's vector.


The code:
  import torch
  from transformers import AutoTokenizer, AutoModel
  from format_data import format_m6_prompt

  def main():
    print("===PHASE 3: EXTRACTING DISTILBERT EMBEDDINGS ===\n")

    # 1. PREPARE THE DATA BATCH
    sample_1 = format_m6_prompt(
        user_history=["clicked blue Nike running shoes", "viewed adidas sports bottle"],
        candidate_item="Puma lightweight training shorts"
    )
    sample_2 = format_m6_prompt(
        user_history=["bought mechanical keyboard", viewed gaming mouse pad"],
        candidate_item="wireless ergonomic mouse"
    )
    batch_prompts = [sample_1, sample_2]

    # 2. LOAD TOKENIZER AND PRETRAINED MODEL
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    # 3. TOKENIZATION
    encoded_inputs = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    print("--- Tokenizer Output Shapes ---")
    print(f"input_ids shape     : {encoded_inputs['input_ids'].shape}")
    print(f"attention_mask shape: {encoded_inputs['attention_mask'].shape}\n")

    # 4. FORWARD PASS THROUGH THE MODEL
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    last_hidden_state = outputs.last_hidden_state

    print("--- MODEL OUTPUT (3D TENSOR) SHAPE ---")
    print(f"last_hidden_state shape: {last_hidden_state.shape}\n")

    # 5. EXTRACT THE [CLS] TOKEN EMBEDDING
    cls_embeddings = last_hidden_state[:, 0, :]

    print("--- EXTRACTED [CLS] EMBEDDINGS (2D TENSOR) SHAPE ---")
    print(f"cls_embeddings shape    : {cls_embeddings.shape}")
    print(f"Simple sample vector    : {cls_embeddings[0].shape}\n")

    print("First 5 features of User 1's [CLS] embeddings:")
    print(cls_embeddings[0][:5])

if __name__ == "__main__":
    main()


    

High Level Map: what does the script actually do?
    1. phase 1 (Setup): you bought your ingredients (Python, PyTorch)
    2. Phase 2 (Formatting): You chopped the vegetables and wrote the recipe on a card (your M6 prompt string)
    3. Phase 3 (this script): -> you put the card into a (super-smart robot chef) (DistilBERT) -> the robot doesn't give you a meal yet.
        instead -> it gives you a scientific analysis (768 numbers) of exactly what is written on that card.

    => These 768 numbers -> are called an "embedding". -> they represent the meaning of your user's history + the candidate item.
        -> we will use these numbers in Phase 4 -> to predict if the user will click.



    Block 1: Importing Libraries (the toolbox)
        import torch
        from transformers import AutoTokenizer, AutoModel
        from format_data import format_m6_prompt

            + (import torch) -> this import PyTorch library. -> PyTorch -> is like a super-powerful calculator -> that does math on Tensors
            (which are just grid of numbers).
            + (from transformers import AutoTokenizer, AutoModel) -> this imports specific tools from HuggingFace.
                + (AutoTokenizer): the tool that turns English text into numbers (we used this in Phase 2)
                + (AutoModel): The tool that loads the actual neural network (DistilBERT) -> that will process those numbers.
            + (from format_data mport format_m6_prompt): this imports the exact function we wrote in phase 2
                -> we just reuse it.


    Block 2: The main() Function and Preparing Data
        def main():
            sample_1 = format_m6_prompt(
                user_history=["clicked blue Nike running shoes", "viewed adidas sports bottle"],
                candidate_item="Puma lightweight training shorts"
            )

                + sample_1 -> this calls your phase 2 function:
                    + what goes in? -> your raw list and the candidate string
                    + what comes out? -> a single string
                    + let's look at the actual value of sample_1 right now:
                        "[BOS] clicked blue Nike running shoes , viewed adidas sports bottle [EOS] [BOS] Puma lightweight training shorts [EOS]"

            sample_2 = format_m6_prompt(
                user_history=["bought mechanical keyboard", "viewed gaming house pad"],
                candidate_item="wireless ergonomic mouse"
            )

                + sample_2 -> now a string
                    "[BOS] bought mechanical keyboard , viewed gaming mouse pad [EOS] [BOS] wireless ergonomic mouse [EOS]"

            batch_prompts = [sample_1, sample_2]
                + batch_prompts -> it's a python list containing exactly 2 strings.
                + Why do we put them in a list? -> neural networks are incredibly fast at doing the exact same math on multiple items at the 
                same time -> this is called parallel processing.
                    -> instead of feeding the model 1 prompt at a time -> we feed a batch (a list) of 2 prompts.
                + value of (batch_prompts) right now: [ <string_1>, <string_2> ]


    Block 3: Loading the Tokenizer and the Model (The Heavy Lifting)
        model_name = "distilbert-base-uncased"
        print(f"Loading '{model_name}' tokenizer and model...\n)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

            + model_name = "distilbert-base-uncased" -> string variable containing the name of the AI model we want to use.
                -> "uncased" -> means it will treat "Nike" and "nike" -> As the same word (lowercases everything).
            
            + tokenizer = AutoTokenizer.from_pretrained(model_name)
                + this goes to the internet (or your cache) -> and downloads the vocabulary for DistilBERT.
                + The tokenizer is like a dictionary -> it knows that the word "clicked" should be turned into the number "1033", and
                "blue" into "13886".

            + model = AutoModel.from_pretrained(model_name)
                + This downloads the actual brain -> for DistilBERT. (268MB of numbers called "weights").
                + this model is a stack of 6 mathematical layers -> it doesn't know english -> it knows pattern matching using billions of
                math operations.


        model.eval()
            + what does (model.eval()) -> do?
                + inside the DistilBERT model -> there's a feature called "Dropout".
                    -> during training -> Dropout randomly turns off some neurons -> to prevent the model from cheating.
                + when we're using the model (not training it) -> we want to turn off Dropout -> so that we get the same predictable numbers
                every time. -> model.eval() -> does that (TURN OFF DROPOUT -> get same predictable numbers).

    
                
    Block 4: Tokenizing the Batch (Turning Strings into Numbers)
        encoded_inputs = tokenizer(
            batch_prompts,      # <-- This is a LIST of 2 strings
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt" #  pt = PyTorch tensors
        )

            -> this is the most critical step.  -> let's examine what happens to our 2 strings:
                + batch_prompts: The tokenizer look at both strings.
                + padding=True:
                    + it counts the tokens in (sample_1) -> let's say it has 48 tokens (words + punctuation + special symbols).
                    + it counts the tokens in (sample_2) -> let's say it has 42 tokens
                    + since they're in the same batch -> they must be the same length -> to form a rectangle
                        -> padding=True says -> find the longest (48 tokens) -> for sample_2 -> add fake [PAD] tokens (which are just 0s)
                            -> at the end until it also has 48 tokens.
                + truncation=True & max_length=64: -> if any prompt is longer than 64 tokens -> just cut it off.
                    (our prompts are ~48 -> nothing is cut)
                + return_tensors='pt': this say -> "return the result as a Pytorch tensor" -> a tensor is just a grid of numbers.

        => What is the actual value of encoded_inputs -> right now?
            -> it's a dictionary (a box with 2 compartments) -> containing 2 tensors:
                1. (encoded_inputs['input_ids']):
                    + this is a 2d tensor with shape (2, 48)
                    + row 1: the 48 token IDs for sample_1
                    + row 2: the 42 token IDs for sample_2
                    + Example of row 1 (first few numbers): [101, 1031, 8945, 2015, 1033,...] -> remember, 101 is [CLS]

                2. (encoded_inputs['attention_mask']):
                    + this is also a 2d tensor with shape (2, 48)
                    + row 1: [1, 1, 1, 1, 1, ... 1] (48 ones, because all tokens are real).
                    + row 2: [1, 1, 1, 1, 1,...1, 0, 0, 0...] -> (42 ones for real words, then 6 zeros for the padding).
                    + the model uses this mask -> to ignore the zeros -> so it doesn't think [PAD] is a real word.
    
    Block 5: The Forward Pass (Running the Brain)
        with torch.no_grad():
            outputs = model(**encoded_inputs)

            + (with torch.no_grad()):
                + in deep learning -> PyTorch usually keeps a diary of every math operation (addition, multiplication) -> so it can calculate
                (gradients) -> for learning later.
                + since we are only extracting numbers -> we don't need this diary.
                    -> turning it off -> saves a lot of memory and makes the code run faster.
            
            + model(**encoded_inputs):
                + The (**) -> is a python trick -> it takes the dictionary (encoded_inputs) -> and unpacks it.
                + instead of writing:
                    model(input_ids=encoded_inputs['input_ids'],
                    attention_mask=encoded_inputs['attention_mask'])
                        -> we just write model(**encoded_inputs)
                            -> the model takes the number, passes them through 6 layers of math, and returns an output.

            + outputs -> this is a special object -> that contains the final numbers.

    Block 6: Extracting the 3D Tensor
        last_hidden_state = outputs.last_hidden_state

            + outputs.last_hidden_state: this is the main result -> it's a 3D tensor.
            => the actual shape of (last_hidden_state) ->  (2, 48, 768)
                + Dimension 0 (2): The batch size (2 prompts).
                + Dimension 1 (48): The sequence length (48 tokens per prompt)
                + Dimension 2 (768): The hidden size. -> DistilBERT uses 768 numbers -> to capture the meaning of each token.
                    =>
                    Imagine it like a binder:
                    You have 2 pages (the 2 prompts).
                        Each page has 48 lines (the tokens).
                            Each line has 768 numbers written on it (the mathematical meaning of that word).


    Block 7: The Magic Slice - Extracting the [CLS] token
        cls_embeddings = last_hidden_state[:, 0, :]
            
            -> the line turns your 3D tensor -> into a 2D tensor.
                -> let's break down the slicing [:, 0, :]:
                    + first colon ":" -> batch dimension -> take all pages (both user 1 and user 2)
                    + The "0" (Sequence dimension) -> take only the first line (index 0) -> on each page.
                        + Which token is at position 0? -> it's the [CLS] token 
                            -> The tokenizer automatically adds [CLS] to the very front.
                        + The [CLS] token -> is special -> because it's trained to summarize the (entire) prompt.
                    + The Second colon ":" -> Hidden dimension -> Take all 768 numbers on that line.

        => What is the actual value of (cls_embeddings) right now?
            + Shape (2, 768):
            + cls_embeddings[0] -> is a 1D tensor of 768 numbers for User 1's prompt.
            + cls_embeddings[1] -> is a 1D tensor of 768 numbers -> for User 2's prompt.

        
        print(f"cls_embeddings shape:   {cls_embeddings.shape}")
        print(f"Single sample vector:   {cls_embeddings[0].shape}")
        print(f"\nFirst 5 features of User 1's embeddings:\n{cls_embeddings[0][:5]}")
            
            + cls_embeddings[0][:5]: This takes User 1's 768 numbers -> and prints only the first 5 (so we don't flood the screen).
            + The numbers look like (tensor([-0.1234, 0.5678, ...])) -> they're just floating points.
                -> the model learned to arrange these numbers -> so that similar user-item pairs -> have similar numbers

                
"""


import torch
from transformers import AutoTokenizer, AutoModel

# Import your Phase 2 prompt formatting function!
from format_data import format_m6_prompt

def main():
    # ----------------------------------------------------
    # Step 1: Prepare Sample Batch Data (M6-Rec Prompts)
    # ----------------------------------------------------
    sample_1 = format_m6_prompt(
        user_history=["clicked blue Nike running shoes", "viewed adidas sports bottle"],
        candidate_item="Puma lightweight training shorts"
    )
    
    sample_2 = format_m6_prompt(
        user_history=["bought mechanical keyboard", "viewed gaming mouse pad"],
        candidate_item="wireless ergonomic mouse"
    )
    
    # We group them into a "batch" of size 2
    batch_prompts = [sample_1, sample_2]

    # ----------------------------------------------------
    # Step 2: Load Tokenizer and Pretrained Model
    # ----------------------------------------------------
    model_name = "distilbert-base-uncased"
    print(f"Loading '{model_name}' tokenizer and model...\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Put the model in evaluation mode (turns off training-specific layers like dropout)
    model.eval()

    # ----------------------------------------------------
    # Step 3: Tokenize the Batch
    # ----------------------------------------------------
    encoded_inputs = tokenizer(
        batch_prompts,
        padding=True,          # Pad the shorter sequence to match the longer one
        truncation=True,       # Cut off if it exceeds max_length
        max_length=64,
        return_tensors="pt"    # pt = PyTorch tensors
    )

    print("=== 1. TOKENIZER OUTPUT SHAPES ===")
    print(f"input_ids shape      : {encoded_inputs['input_ids'].shape}")
    print(f"attention_mask shape : {encoded_inputs['attention_mask'].shape}\n")

    # ----------------------------------------------------
    # Step 4: Forward Pass Through the Model
    # ----------------------------------------------------
    # torch.no_grad() tells PyTorch NOT to track math for backpropagation, saving memory
    with torch.no_grad():
        # **encoded_inputs unpacks the dictionary so it passes input_ids=... and attention_mask=...
        outputs = model(**encoded_inputs)

    # Extract the raw 3D tensor from the model's output
    last_hidden_state = outputs.last_hidden_state

    print("=== 2. MODEL OUTPUT (3D TENSOR) SHAPE ===")
    print(f"last_hidden_state shape: {last_hidden_state.shape}\n")

    # ----------------------------------------------------
    # Step 5: Extract the [CLS] Token Embedding
    # ----------------------------------------------------
    # We slice the 3D tensor to get just the first token (index 0) of every sequence in the batch.
    cls_embeddings = last_hidden_state[:, 0, :]

    print("=== 3. EXTRACTED [CLS] EMBEDDINGS SHAPE ===")
    print(f"cls_embeddings shape   : {cls_embeddings.shape}")
    print(f"Single sample vector   : {cls_embeddings[0].shape}")
    print(f"\nFirst 5 features of User 1's embedding:\n{cls_embeddings[0][:5]}")

if __name__ == "__main__":
    main()

