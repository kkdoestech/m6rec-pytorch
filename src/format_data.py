'''
A neural networks (like DistilBERT) -> does not understand raw English. It only understands numbers (tensors).
  so we must:
    1. Convert text into a structed string (the prompt).
    2. Convert that string -> into a list of numbers (input_ids) using a tokenizer.
    3. Give the network these numbers + an (attention_mask) -> so it knows which numbers are actual words vs. padding.


  Step 1: The M6-Style Prompt
    -> in the M6-Rec paper -> they treat user history and candidate items -> as sequences of text. -> they use special markers to tell the
    model: "Here ends the history, here begins the candidate item".
      you wrote: [BOS'] User history text... [EOS'] [BOS] Candidate item text... [EOS]
        
        -> DistilBERT does NOT have [BOS] or [EOS] tokens -> it was trained with [CLS] and [SEP].
          But that's totally fine! We can still put the literal strings "[BOS]" and "[EOS]" into our prompt. 
            -> The tokenizer will just treat them as normal words (it will split them into subwords). 
              -> The model will learn to associate these symbols with boundaries during training.

    
    Our final prompt string will look like this:
      "[BOS] clicked blue Nike running shoes , viewed adidas sports bottle [EOS] [BOS] Puma lightweight training shorts [EOS]"



  Step 2: Tokenization - Turning Text into Numbers
    -> Tokenization is the process of splitting text into smaller piece (tokens) and mapping each piece to a unique integer ID.
      + AutoTokenizer.from_pretrained("distilbert-base-uncased") -> loads the exact same tokenizer that was used to train DistilBERT.
        -> it knows:
          + The vocabulary (a dictionary: {"hello": 1234, "world": 5678, ...})
          + Which specific tokens -> to add automatically.

    
    Special tokens added by DistilBERT (automatically):
      + [CLS] (token id 101) -> inserted at the very beginning of every sequence. -> For classification tasks, the final output for this token is used as the
      overall SENTENCE REPRESENTATION.
      + [SEP] (token id 102) -> inserted BETWEEN sentences and at the end. -> it tells the model -> "SENTENCE BOUNDARY" here.
      + [PAD] (token id 0) -> used to make all sequences the same length -> in a batch.

    Important:
      Even though we put [BOS] and [EOS] in our custom prompt, the tokenizer will add [CLS] at the very front and [SEP] at the very back automatically 
      (unless we tell it not to). So the final sequence actually becomes:
        [CLS] [BOS] user history ... [EOS] [BOS] candidate ... [EOS] [SEP]

        
        
  Step 3: The Output Tensors
    -> when we call the tokenizer, we get a dictionary containing:
      + input_ids -> a list (or tensor) of integers. Each integer is the ID of a token. 
        example:
          [101, 456, 789, ..., 102]
      + attention_mask -> a list of 1s and 0s, same length as input_ids.
        + 1 -> means "pay attention to this token" (real text).
        + 0 -> means "ignore this token" (it's just padding to make the length equal across samples).
      + token_type_ids (optional) -> tells the model which sentence each token belongs to (first sentence vs second).
        -> not critical for our single-string use case.

        


  Step 4: The Python Function
    Load the tokenizer, formatted the prompt -> and converted it into numbers -> so a neural network can understand.

      The (format_m6_prompt) function:
        def format_m6_prompt(user_history, candidate_item):
          history_text = " , ".join(user_history)
          prompt = f"[BOS] {history_text} [EOS] [BOS] {candidate_item} [EOS]"
          return prompt

            + user_history -> is a list containing 2 strings:
              ["clicked blue Nike running shoes", "viewed adidas sports bottle"]
            + candidate_item -> is a single string:
              "Puma lightweight traning shorts"

            Line 1: history_text = " , ".join(user_history)

              + the .join() method -> glues all the items in the list together -> into 1 big string.
              + the " , " -> is the glue -> it goes between each item

              -> so your list becomes:
                "clicked blue Nike running shoes" , viewed adidas sports bottle"
                (the comma and spaces are added automatically).

            Line 2: prompt = f"[BOS] {history_text} [EOS] [BOS] {candidate_item} [EOS]"

              + The f before the quotes makes it an fstring -> it lets you put variables directly inside {} braces.
              + it inserts (history_text) and (candidate_item) into the template.
              + the final (prompt) string becomes:
                "[BOS] clicked blue Nike running shoes , viewed adidas sports bottle [EOS] [BOS] Puma lightweight training shorts [EOS]"
                (this is what the output printed)
    
                
      The (tokenize_m6_prompt) function:
        def tokenize_m6_prompt(prompt, tokenizer, max_length=128):
          encoded = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
          )
          return encoded

            + Line 1: encoded = tokenizer (...) -> calls the tokenizer object.
              -> let's break down every argument:
                + (prompt): the text string you want to convert -> your m6 prompt. (In your run (max_length=64))
                + (truncation=True): if the text is longer than (max_length) -> cut it off at that limit
                  -> Your prompt is about 35 tokens long, so nothing was cut off.
                + (padding='max_length'): if the text shorter than (max_length) -> add fake [PAD] tokens to the end until it reaches
                exactly (max_length)
                  -> Your prompt is 35 tokens, so the tokenizer added 29 [PAD] tokens (64-35=29) to make the length exactly 64.
                + (max_length=64) -> the target length -> every output will have exactly 64 numbers.
                  -> we get 64.
                + (return_tensors='pt') -> return the result -> as Pytorch tensors (data type for neural networks).
                  -> the outputis a tensor of shape (1, 64) -> meaning 1 batch, 64 tokens.

    What does (encoded) contain?
      it's a dictionary (like a little box with labelled compartments) with two keys:
        1. 'input_ids' -> a tensor of token IDs (integers)
        2. 'attention_mask' -> a tensor of 1s and 0s.



    

        
The (if __name__ == "__main__") block:
  -> python trick so the code runs only when you execute the file directly.

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
      
      + this loads the tokenizer that was specifically trained -> for the DistilBERT model.
      + it downloads a few small files (config.json, vocab.txt, tokenizer.json) -> these contain the vocabulary (a giant mapping
      words -> IDs)
      + The "uncased" means it lowercases everything (Nike -> nike, ...)

    prompt = format_m6_prompt(user_history, candidate)
      
      + call the function -> and stores the formatted string.

    encoded = tokenize_m6_prompt(prompt, tokenizer, max_length=64)

      + call the tokenizer -> and stores the dictionary with tensors.

    print(encoded['input_ids'][0][:20])

      + encoded['input_ids'] -> gets the tensor of IDs.
      + [0] -> get the first sequence of batch
      + [:20] -> get the first 20 token IDs (so we don't print all 64 and clutter the screen)
      + this prints the tensor you see in the output.

    print(encoded['attention_mask'][0][:20])
      + same as above, but for the mask

    first_20_ids = encoded['input_ids'][0][:20]
    decoded = tokenizer.decode(first_20_ids)
    print(decoded)
      
      + tokenizer.decode() -> take the IDs and turns them back into text -> so you can see what the tokenizer actually "read".
        -> this is a great debugging tool.



  
        
Part 2: Explaining Your Output Numbers
  input_ids (first 20 tokens):
  tensor([  101,  1031,  8945,  2015,  1033, 13886,  2630, 18368,  2770,  6007,
            1010,  7021, 27133,  8883,  2998,  5835,  1031,  1041,  2891,  1033])

            
    What are these numbers?

        Every number is the unique ID of a token in the tokenizer's vocabulary.

        Let's decode the first few to see what they represent:
        Token ID	Actual text (decoded)
        101	[CLS] (automatically added)
        1031	[ (the opening bracket)
        8945	bos (lowercased)
        2015	] (closing bracket)
        1033	clicked
        13886	blue
        2630	nike
        18368	running
        2770	shoes
        6007	, (comma)
        1010	viewed
        7021	adidas
        27133	sports
        8883	bottle
        2998	[
        5835	eos
        1031	]


          [CLS] (ID 101) was added automatically - you didn't type it, but the tokenizer inserted it at the very front.

          [BOS] and [EOS] became [, bos, ] and [, eos, ] because the tokenizer is uncased (lowercases everything) and 
          splits punctuation into separate tokens.
              
          
  The attention_mask:

    attention_mask (first 20 tokens):
    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        All 1s for the first 20 tokens - because these are real words.

        If we printed the full 64-length mask, you would see:

            Positions 0-34: all 1s (real words up to the 35th token).

            Positions 35-63: all 0s (these are the [PAD] tokens added to reach length 64).

    Why does the mask matter?
    The neural network uses the mask to ignore the padded zeros. Without the mask, it would think [PAD] is a real word like 
    "the" and get confused.

    



    The decoded text:

      [CLS] [ bos ] clicked blue nike running shoes, viewed adidas sports bottle [ eos ]

        This is the first 20 tokens turned back into words.

        Notice it stops at [ eos ] - that's because the second half of your prompt (the candidate item) starts after that, 
        at token position 21.

        The full decoded text (if you decoded all 64 tokens) would be:
        [CLS] [ bos ] clicked blue nike running shoes, viewed adidas sports bottle [ eos ] [ bos ] puma lightweight training shorts 
        [ eos ] [PAD] [PAD] ... [PAD]

  
        

  
'''

from transformers import AutoTokenizer

def format_m6_prompt(user_history, candidate_item):
    """
    Convert raw history and a candidate item into an M6-style text prompt.

    Args:
      user_history (list of str): 
        eg: ["clicked blue Nike running shoes", "viewed adidas sports bottle"]

      candidate_item (str):
        eg: "Puma lightweight training shorts"

    
    Returns:
      str: The formatted prompt string.
    """
    # 1. Join the history items into a single string, seperated by commas.
    #    e.g., "clicked blue Nike running shoes , viewed adidas sports bottle"
    history_text = " , ".join(user_history)

    # 2. Build the prompt using the M6-style delimiters.
    #  We use [BOS] and [EOS] as literal substrings.
    prompt = f"[BOS] {history_text} [EOS] [BOS] {candidate_item} [EOS]"

    return prompt

def tokenize_m6_prompt(prompt, tokenizer, max_length=128):
    """
    Tokenizes the prompt using the provided tokenizer and returns PyTorch tensors.

    Args:
      prompt (str): The formatted prompt string.
      tokenizer: A hugging face AutoTokenizer instance.
      max_length (int): Maximum sequence length (truncate if longer, pad if shorter).

    Returns:
      dict: A dictionary with 'input_ids', 'attention_mask' (and optionally 'token_type_ids').
            Each value is a Pytorch tensor.
    """
    # Tokenize the prompt.
    # Parameter explained:
    #   + truncation=True: if the prompt is longer than max_length -> cut it off.
    #   + padding='max_length': add [PAD] tokens at the end -> to reach exactly max_length.
    #   + max_length=max_length: set the target length.
    #   + return_tensors='pt': return Pytorch tensors (pt=Pytorch).
    #   + add_special_tokens=True (default): automatically adds [CLS] at start and [SEP] at end.
    encoded = tokenizer(
        prompt,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'  # Pytorch tensors
    )

    return encoded

# ============ Let's test it with your sample data ============
if __name__ == "__main__":
    # Sample raw data
    user_history = ["clicked blue Nike running shoes", "viewed adidas sports bottle"]
    candidate = "Puma lightweight training shorts"

    # 1. Load the tokenizer (same as we use in phase 1)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 2. Format the script
    prompt = format_m6_prompt(user_history, candidate)
    print("Formatted Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")

    # 3. Tokenize it
    encoded = tokenize_m6_prompt(prompt, tokenizer, max_length=64)

    # 4. Let's inspect the output
    print("input_ids (first 20 tokens):")
    print(encoded['input_ids'][0][:20])  # [0] because batch size = 1
    print("\n attention_mask (first 20 tokens):")
    print(encoded['attention_mask'][0][:20])

    # 5. Decode the first 20 IDs back to text -> to see what the tokenizer actually sees.
    print("\n Decoded first 20 tokens:")
    first_20_ids = encoded['input_ids'][0][:20]
    decoded = tokenizer.decode(first_20_ids)
    print(decoded)

    # 6. Show the special tokens' IDs
    print("\n Special token IDs:")
    print(f" [CLS] id = {tokenizer.cls_token_id}")
    print(f" [SEP] id = {tokenizer.sep_token_id}")
    print(f" [PAD] id = {tokenizer.pad_token_id}")

