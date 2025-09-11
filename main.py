import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT(nn.Module):

    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    self.key_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.query_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.value_layer = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.key_layer(embedded)
                    q = self.query_layer(embedded)
                    v = self.value_layer(embedded)

                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = (lower_triangular == 0).to(device)
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v

            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                self.attention_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.attention_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.compute = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, embedded):
                head_outputs = []
                for head in self.attention_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.dropout(self.compute(concatenated))

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2

            def forward(self, x):
                return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.vanilla_nn = self.VanillaNeuralNetwork(model_dim)
            self.layer_norm_one = nn.LayerNorm(model_dim)
            self.layer_norm_two = nn.LayerNorm(model_dim)

        def forward(self, embedded):
            embedded = embedded + self.mhsa(self.layer_norm_one(embedded)) # skip connection
            embedded = embedded + self.vanilla_nn(self.layer_norm_two(embedded)) # another skip connection
            return embedded

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.layer_norm_three = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        embedded = self.token_embedding(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length).to(device)
        embedded = embedded + self.pos_embedding(positions)

        raw_output = self.vocab_projection(self.layer_norm_three(self.transformer_blocks(embedded)))
        # raw_output is batch by context_length by vocab_size

        return raw_output


def generate(model, new_chars: int, context, context_length: int, int_to_char: dict) -> str:
    res = []
    for i in range(new_chars):
        if len(context.T) > context_length:
            context = context[:, -context_length:]
        prediction = model(context) # B, T, Vocab_Size
        last_time_step = prediction[:, -1, :] # B, Vocab_Size
        probabilities = nn.functional.softmax(last_time_step, dim = -1)
        next_char = torch.multinomial(probabilities, 1)
        context = torch.cat((context, next_char), dim = -1)
        res.append(int_to_char[next_char.item()])
    return ''.join(res)


import os
import subprocess

# Clone the repo if not already cloned
if not os.path.exists("drake-lyric-generator"):
    subprocess.run(["git", "clone", "https://github.com/gptandchill/drake-lyric-generator"])

# Change directory
os.chdir("drake-lyric-generator")

print("Now inside:", os.getcwd())



vocab_size = 104
context_length = 128
model_dim = 252
num_blocks = 6
num_heads = 6

model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads).to(device)
WEIGHT_PATH = 'weights.pt' # Adjust as necessary
model.load_state_dict(torch.load(WEIGHT_PATH,map_location=device))
model.eval()
new_chars = 5000
context = torch.zeros(1, 1, dtype = torch.int64).to(device)

int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}

print(generate(model, new_chars,context,
               context_length,
               int_to_char))

from sacrebleu import corpus_bleu

# hypotheses = list of generated strings
# references = list of lists, where each inner list contains references for the same hypothesis
# Example:
hypotheses = ["i started from the bottom now we here"]
references = [["i started at the bottom now we here", 
               "we started at the bottom now we're here"]]

# sacrebleu expects refs grouped by reference index, so we transpose
refs = list(map(list, zip(*references)))

bleu = corpus_bleu(hypotheses, refs)
print("BLEU score:", bleu.score)
