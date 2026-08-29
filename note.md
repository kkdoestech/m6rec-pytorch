In the Big app (like Taobao), the system isn't just one thing. There is a pipeline.
  + Retrieval: Looking through 100 million products to find 100 possible matches (like a librarian grabbing 100 books off a billion-book
  shelf.)
  + Ranking: Taking those 100 and ordering them from "most likely to click" to "least likely" (like a judge ranking the books).
  + Generation: Writing a review, explaining why you liked it, or even designing a new product title.

The current way (Bad): Build a seperate AI for each of these -> That means training 100 different models.



-> A "Foundation" Model here is a Transformer that has been pre-trained on internet text (like Wikipedia, News, etc).
  -> Why "text": They convert everything (user history, product details) -> into a string of English words.


-> The "Tight Hardware Budget" -> They want to adapt this giant model to new tasks without retraining the whole thing.
  -> They use Prompt Tuning (changing just the input prompt) rather than Fine-Tuning (Changing the whole AI's weights).


Part 2: The Introduction (Deep Dive) - "Why should we care"?
  The "Bias Reduction" Argument:
    -> If you make the "Ranking" AI better -> the "Retrieval" AI might still feed it bad data.
      -> If you optimize one AI for the whole pipeline (M6-Rec) -> improving the foundation improves everything.

  Zero/Few-Shot Learning:
    -> "A male user in Beijing, who clicked product X last night... was recommended product Z and did not click it."
      -> Traditional AI uses User ID 4821 and ProductId 9032 -> if the AI has never seen ID 9032 -> it fails.
        -> By using text ->the AI reads "hiking shoes" and "winter jacket" -> even if the AI has never seen that specific product before
          -> it has read Wikkipedia about hiking and jackets. -> it uses it own knowledge to guess the user's intent.
            -> This is core of Zero-Shot (making a guess with zero specific training data).


Part 3: Related Work (Why everyone else is wrong, and M6-Rec is right)
  -> 3 failuresof old systems:
    1. Old systems ignored the Web:
      + old way: only look at user clicks (behaviour data).
      + m6-rec way: -> M6 was pre-trained on the web. -> so if Thanksgiving is coming -> M6 knows people eat turkey -> even if no one in 
      the app has clicked on turkey yet.
        => technical takeaway: they are fusing Behavioral Modality (clicks) with Linguistic Modality (text meaning).

    2. Old systems used ItemIds:
      + old way: Product_ID = [0, 0, 1, 0, 0] (one-hot encoding).
      + M6-rec way: Product = "waterproof hiking shoes" (text tokens).
        -> this means M6-rec -> can recommend products that didn't exist when the AI was first trained (Open-Domain recommendation).

    3. Old systems only gave Scores (Classification):
      + Old way: Output a number between 0 and 1 (will click? Yes/No).
      + M6-rec way: -> because it's based on GPT-liked models -> it can generate sentences.
        -> this allows "Explanation Generation" (writing why you recommend it).


Related work (Efficiency):
  -> First, think of the Transformer as a 24-story building (24 layers).
    + Early Exiting: if the AI is 99% sure at floor 2 -> it jumps out the window early to save time.
    + Pruning: Cutting out 80% of the wires inside the AI that don't do much.
    + Quantization: Changing numbers from 32-bit decimals to 8-bit decimals (takes less memory).



Part 4: The Method - Section 3 (The Hardcore Mechanics).
  M6 - The Backbone:
    M6 - base consists of a single Transformer of L = 24 layers, H = 16 attention heads, and d = 1024 hidden states.
      + L=24: 24 layers deep.
      + H=16: 16 attention heads (-> meaning: it looks at 16 different relationships between words simultaneously).
      + d=1024: The vector size (Every piece of text -> is converted into a list of 1024 numbers).


  Figure 1: The Text Infilling Objective (Pretraining)
    -> Look at the figure description: [MASK] represents unknown tokens.
      + how it learns: the AI is given a sentence with a mask: "The user bought [MASK] shoes".
      + The Bidirectional Region (left side) -> reads the whole sentence forwards and backwards to understand context.
      + The Autoregressive Region (right side) -> predicts the masked word one by one (like GPT).
        -> it says [MASK] = "hiking"
      + this dual-training -> makes it perfect for understanding (BERT-style) and generating (GPT-Style).


  3.2. -> Behavior Modeling as Language Modeling (The Core Data Format)
    -> most important part of the paper

      For Scoring Tasks (CTR Prediction):
        -> look at the massive text prompt:
        [BOS'] December. Beijing... A male user... clicked jacket... [EOS'] [BOS] The user is now recommended boots... [EOS]

          + [BOS'] and [EOS'] wrap the User's history. -> this part uses the Bidirectional attention mask.
            -> Why? because when the AI reads the user's history -> it needs to see all past clicks at the same time -> to understand the
            user's overall taste.
            
          + [BOS] and [EOS] wrap the Candidate item (the boots). -> this uses the Autoregressive (left-to-right) mask.
            -> Why? because the AI is reading the candidate item and predicting the final outcome (click or not) at the very end.

          + At the [EOS] token of the candidate -> the AI spits out a hidden vector. -> this vector goes into a "Linear Softmax Classifier"
          (basically, Vector * Weights = Score). -> if the score is high -> click; if low -> no click.


      For Generation Tasks (Explanation/Product Design):
        -> They just give the prompt up to the word "because" and let the AI autocomplete (generate) the rest of the sentence.
          -> this uses the autoregressive loss just like chatgpt.


      Zero-Shot Scoring (The Coolest Trick):
        -> they don't train a classifier. -> instead they write 2 sentences:
          1. "user clicks hiking shoes [and] clicks trekking poles"
          2. "user clicks hiking shoes [and] clicks yoga knee pads".
            -> they ask M6: "Which sentence is more probable based on your internet knowledge?"
              -> They compare log(p1) + log(p2) (likelihood of the words) -> The AI will naturally say "trekking poles" is more likely
              because outdoorsmen buy both.
                -> This requires ZERO training data.

      Retrieval Tasks (Vector Search):
        + They feed the user's text into a model -> and take the hidden state at [EOS'] -> They pass it through a linear layer -> to squash
        it down to 128 dimensions.
        + They do the same for the item
        + They use Contrastive Learning (the scary formula with (exp) and (log))
          + The intuition: Pull the User vector and Item vector closer together -> if the user bought it (Positive Pair). Push them apart
          if the user didn't (Negative Pairs).
          + 12-normaliztion -> just means -> they scale the vector to a length of exactly 1 -> so they can quickly search millions of items
          using dot products.


Part 5: Efficiency Tricks - section 3.3 & 3.4
  Figure 3: Multi-Segment Late Interaction
    -> The Problem: A user has 100 past clicks. running the 24-layer AI on all 100 clicks for every millisecond request -> is Impossible.

      -> The Solution:
        + They chunk the users's history into segments: [Segment 1: Male], [Segment 2: Clicked X], [Segment 3: Clicked Y].
        + They run only the first 21 layers on these segments offline (before the user even opens the app) and save the outputs in a cache
        (like a database).
        + When the user actually requests a recommendation -> they just grab these saved outputs, glue them together -> and run only the last
        3 layers (which is very fast).
        + Crucial detail: They add a special "Segment Position Enbeddings" -> so the last 3 layers -> know which token belongs to Segment 1
        versus Segment 2.


  Figure 2 & Section 3.4: Option-Adapater Tuning
    + Normal Fine-Tuning: you adjust all 300 million weights. -> Huge memory waste.
    
    + Prompt Tuning: you add 100 "fake words" (soft prompts) -> to the start of your output.
      -> you only train these 100 fake words.
        -> The Problem with Prompt Tuning: it trains slowly.
    
    + Their "Option Tuning" Fix: -> instead of making a seperate classifier head (a new small neural networks) -> they take the last few soft
    prompts -> and directly use their values -> as the weights for the final decision (like "Yes" or "No")
      => This is like giving the AI multiple-choice options embedded in the prompt itself, making it converge (learn) much faster.

    + They add Adapters (tiny 2-layer neural networks inserted between layers) -> Despite only training 1% of the model's parameters
      -> they beat full fine-tuning.

    

=============================================================================================================================================









Part 1: The Engine Specs (Section 3.1 - M6 Backbone)
  -> M6-base => consists of a single Transformer of L=24 layers, H=16 attention heads, and d=1024 hidden states.
    -> when the AI reads a sentence -> every word gets turned into a list of 1024 numbers (a vector). 
      -> that list pass through 24 "processing floors" (layers). on each floor -> 16 different "detectives" (attention heads) look at the
      relationships between words.

  The "UniLM" Attention Mask: Look at Figure 1
    + Left side (Bidirectional): The user history [BOS']...[EOS'] -> the AI can see past and future words in this section simultaneously.
      (Like reading a whole paragraph and understanding context).

    + Right Side (Autoregressive): The candidate item [BOS]...[EOS] -> the AI can only see left-to-right (like chatGPT).
      -> Why? because when generating text or predicting a click -> it predicts one word at a time based on the previous words.


Pretraining Tasks:
  1. Text infilling: they hide a few words [MASK] -> and force the AI to guess them. 
    -> this teaches the AI logic (scoring plausibility).
  
  2. Autoregressive Generation: they hide the entire second half and force the AI to write it
    -> This teaches the AI fluency.




Part 2: The Data Format (Section 3.2 - The Magic String)
  -> this is the most important concept: They kill the "UserId" and "ItemId"
   
   For Scoring (CTR Prediction):
     ex: [BOS'] December. Beijing... A male user clicked jacket... [EOS'] [BOS] The user is now recommended boots... [EOS]
       + Your takeaway: They are literally concatenating strings.
       + The [BOS'] to [EOS'] -> is the User's History (Bidirectional, so the AI reads the whole shopping history to understand the user).
       + The [BOS] to [EOS] is the Candidate's Story (Autoregressive, so the AI reads the product details and predicts the final output
       at the very last token [EOS])
       + They take the hidden vector at the final [EOS], shove it into a Linear(1024 -> 2) layer (a simple mathematical matrix multiplication), and use Cross-Entropy Loss (the standard classification loss you learn in class) -> to decide "Click" or "No Click"

    
    For Generation (Explanations):
      -> they feed the prompt up to the word "because" -> and let the AI autocomplete. The loss used here is Autoregressive Language Modeling
      Loss (Predicting the next token, Token_1 -> Token_2 -> Token_3) -> this is how ChatGPT generates text.

    For Zero-Shot Scoring (The Smartest Hack):
      -> They don't train a classifier. Instead, they ask the AI: "Which sentence is more likely?"
        -> Formula: InfoNCE Loss (Standard for AI in 2024)
            "...minimizing ∑−log⁡exp⁡(xTy/τ)exp⁡(xTy/τ)+∑exp⁡(xTy′/τ)"

              + x = The User Vector (a list of 128 numbers).
              + y = The Item vector of the product user actually clicked (Positive).
              + y' = A bunch of random items the user did not click (Negatives).
              + x^T y = The Dot Product -> In Linear Algebra -> if 2 vectors point in the same direction -> the dot product is large
                -> we want x and y -> point in the same direction.
              + exp(... / tau) = exponentiates the dot product -> to make big numbers bigger, small numbers smaller (tau=0.07) -> is just 
              a temperature knot.

        => What the loss Does: it forces the numerator (User vs Clicked item) -> to be as huge as possible, and forces the denominator (User
        vs Random Items) -> to be as tiny as possible.
          -> By minimizing this -log function -> the model learns to pull x and y together -> in the 128-dimensional space.


    
Part 3: Making it FAST (Section 3.3 - Multi-Segment Late Interaction)
  The problem: 24 layers is slow. if a user has 100 past clicks -> processing all 100 thorough 24 layers for every request -> takes ~57 
  milliseconds (table 6).
    -> The Solution (Caching):
      + They take the User's history ("Clicked X", "Clicked Y") -> and run only the first 21 layers on these segments offline (before the
      user even opens the app).
      + They save (cache) the output of layer 21 in a database.
      + When the user actually requests a recommendation, the server just grabs these pre-computed layer-21 outputs, glues them together, and runs ONLY the last 3 layers.

    + Position Enbeddings: Since each segment was processed alone -> the last 3 layers don't know which token belongs to "Clicked X" vs 
    "Clicked Y" -> so they add a "Segment ID" embedding (0 for segment 1, 1 for segment 2) to the input -> just like how normal Transformers add "Word Position" embeddings.



Part 4: The Efficient Training Hack (Section 3.4 - Option Tuning)
  -> Normally, to adapt this 300M model to a new task -> you'd do Fine tuning (update all 300M weights).
    -> That's expensive. They use Prompt Tuning (only update 100 fake words added to the input). -> But Prompt Tuning learns slowly.

  Their "Option Tuning" fix:
    Instead of training a new (Linear) layer for the final decision (Click/No Click) -> they take the last few soft prompts -> and literally 
    use their vectors -> as the classifier weights.
      (think of it like giving the AI a multiple-choice answer sheet built right into the question). This converges faster.



  The Adapter Formula (This is pure Linear Algebra):
    FFN(t)(Z)=FFN(t)(Z) + λ(t)⋅[σ(Z.W1​+b1​).W2​ + b2​]
      
      + Z = the input matrix. Shape is [Batch_Size (B) x Hidden_Size (d)]. Let's say B=32, d=1024
      + W1 -> is a matrix of shape [d x r], r is a tiny number, like 8.
      + Z.W1 -> multiplies [32 x 1024] by [1024 x 8] -> resulting in [32 x 8] -> this shrinks the data down.
      + sigma -> is just ReLU (an activation function, max(0, x))
      + W2 ->is a matrix of shape [r x d] (8 x 1024) -> multiplying [32 x 8] by [8 x 1024] -> gives [32 x 1024]
        -> this expands it back to the original size.
      + Why do this? -> this tiny "bottleneck" (size 8) -> is the only thing they train
        -> it acts like a sticky note inserted into the AI. they train only W1, W2 and lambda (1 % of the total parameters), but because
        this tiny detour is placed INSIDE EVERY LAYER -> it has enough power to beat full fine-tuning.



Part 5: Putting it on your Phone (Section 3.5 - M6-Edge)
  300M parameters is 1.2GB of memory. That kills your phone battery. so they compress it brutally:
    1. Distillation (300M -> 10M): They train a small "student" AI -> to mimic the outputs of the big "teacher" AI.
    2. Parameter Sharing (ALBERT style): They make all 24 layers -> use the exact same weights. 
      -> instead of 24 different sets of weights, it's 1 set reused 24 times. -> this drastically shrinks memory.
    3. Pruning (10M -> 2M): They look at all the weights in the model. If a weight is close to 0 -> it's useless.
      -> They force 80% of them to become exactly 0 (removing connections).
    4. Early Exiting: During inference, if your phone is slow -> the AI stops at layer 2 instead of layer 24.
      -> They optimize the loss as ∑ (2k / (k*(k+1)) ) * ​Lk​ - which is just a fancy weighted average -> that forces all early layers to be
      useful, not just the final one.
    


Part 6: The Results (Section 4 - What the Table tell you)
  + Table 1 (Explanation):
    -> look at the M6-Rec row.
      -> (ROUGE-F) (quality of generated text) -> goes from ~15 to 34. (DIV) (diversity of text) -> goes down to 0.89 (meaning it doesn't
      repeat itself, good!).
        => They crushed the previous state-of-the-art (PETER+).

  + Table 2 & 3 (The "Unseen" Superpower):
    + YoutubeDNN uses ID embeddings. When a new item appears that wasn't in training -> it literally has no ID vectors. -> it fails (0%).
    + M6-Rec uses Text. A new item like "Nike Air Max 2025" -> still has text.
      -> the AI understands "Nike" and "Air" -> M6-Rec score 57.0% -> this is the biggest advantage.

  + Table 6 (Latency):
    + Full 24 layers: 57ms
    + Late Interaction (caching 21 layers, running 3): 16ms. (Same speed as a tiny 3-layer model, but with almost the same accuracy as the
    full 24-layer model).

  + Table 7 (Edge):
    + They got a 2 million parameter model (M6-Edge-Pruned) -> that scores (0.537) on TNEWS.
      it's worse than the 10M version (0.552) -> but it's small enough to run on a low-end phone -> without crashing the app.




  