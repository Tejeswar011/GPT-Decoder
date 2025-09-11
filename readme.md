# GPT Decoder from Scratch

This project is a **from-scratch GPT-style decoder** implemented in PyTorch, following the classic design of the *Attention Is All You Need* paper (2017).  
The goal is to provide an **educational, minimal yet functional** implementation that can generate coherent sequences like lyrics or text character-by-character.  

---

##  Features
- Multi-Head Self-Attention with causal masking  
- Standard feed-forward layers (2-layer MLP with ReLU)  
- Learned token + positional embeddings  
- Residual connections + Layer Normalization  
- Autoregressive text generation  
- Implemented entirely with **PyTorch** from first principles  

---

##  Decoder Blocks Explained in Detail

### 1. **Token Embeddings**
Raw text is first converted into numerical IDs (e.g., “A” = 29, “b” = 59). The **token embedding layer** maps each ID to a dense vector of size `model_dim`. This step is crucial because neural networks work with continuous values, not discrete integers. Over training, embeddings learn semantic structure — for instance, the vectors for similar characters or words will end up close in space. In effect, embeddings turn a lookup table into a learnable semantic space that provides the building blocks for attention.

---

### 2. **Positional Embeddings**
Unlike RNNs, Transformers have no built-in notion of order. Self-attention treats inputs like a set, so we need a way to inject positional information. **Positional embeddings** are learned vectors that correspond to sequence positions (0, 1, 2 … up to context length). Each token’s embedding is added to its position embedding, so the model knows not only “what” the token is, but also “where” it appears. This is how the model understands sequence structure like left-to-right word order or rhyme placement in lyrics.

---

### 3. **Multi-Head Self-Attention (MHSA)**
The heart of the decoder is self-attention. Each token embedding is projected into **queries (Q)**, **keys (K)**, and **values (V)**. Queries compare against keys to calculate attention scores, which decide how much each token should focus on others. Scores are normalized with softmax and used to weight the values, producing a context-aware representation. Multiple heads run in parallel, letting the model capture different types of dependencies (syntax, repetition, rhythm) at once. A **causal mask** ensures that a token can only attend to itself and previous tokens — never future ones. This is what makes the model autoregressive and capable of generating text step by step.

---

### 4. **Vanilla Feed-Forward Network**
After attention, each token passes through a **two-layer feed-forward network** with a ReLU activation. The first layer expands the dimension (typically 4× `model_dim`), and the second layer projects it back. This non-linear transformation gives the model more representational power, allowing it to mix and remap information captured by attention. While attention finds relationships between tokens, the feed-forward network processes each token independently to enrich its representation. Dropout is added to reduce overfitting.

---

### 5. **Residual Connections & Layer Normalization**
Every sub-block (attention and feed-forward) is wrapped with **residual (skip) connections** and **layer normalization**. Residuals allow the model to pass information forward unchanged if needed, preventing loss of key details and helping gradients flow. Layer normalization stabilizes training by re-centering and scaling activations, keeping learning stable across layers. Together, they solve major optimization problems like vanishing/exploding gradients and make deep Transformers trainable.

---

### 6. **Final Projection Layer**
After passing through multiple stacked decoder blocks, the output is normalized one last time and sent through a **linear projection** that maps hidden states to vocabulary size. This produces logits (unnormalized probabilities) over the entire vocabulary at each time step. During training, these logits are compared with the ground-truth next token using cross-entropy loss. During generation, a softmax is applied to sample the next token, which is then appended to the sequence. This step closes the loop for autoregressive generation.

---

##  Example Usage
```python
model = GPT(vocab_size=104, context_length=128, model_dim=252, num_blocks=6, num_heads=6).to(device)
model.load_state_dict(torch.load("weights.pt", map_location=device))
model.eval()

context = torch.zeros(1, 1, dtype=torch.int64).to(device)
print(generate(model, 500, context, 128, int_to_char))
