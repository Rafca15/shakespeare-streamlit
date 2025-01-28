import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    # one head of self attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)    # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention score ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # multiple heads of self-attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    # a simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # transformer block: communication followed by computation

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

st.title("Shakespeare Chatbot")

device = 'cpu'
vocab_size = 65  # Replace with your vocab_size
block_size = 128
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

model = BigramLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
model.load_state_dict(torch.load("shakespeare_transformer.pth", map_location=device))
model.eval()

chars = sorted(list(set(open("input.txt", "r", encoding="utf-8").read())))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.text_input("Enter your text in modern English:", key="user_input")

if st.button("Send"):
    if user_input:
        # Append user input to conversation
        st.session_state.conversation.append({"role": "user", "text": user_input})

        # Use only the most recent user input as context
        context_text = user_input  # Only use the latest input
        context = torch.tensor([encode(context_text)], dtype=torch.long, device=device)

        # Ensure context does not exceed block_size
        if context.shape[1] > block_size:
            context = context[:, -block_size:]  # Truncate to block_size

        # Generate chatbot response
        response_idx = model.generate(context, max_new_tokens=50)[0].tolist()  # Reduced max_new_tokens for brevity
        response = decode(response_idx).strip()

        # Clean up the response (remove unwanted characters or formatting)
        response = response.split("\n")[0]  # Take only the first line of the response
        response = response.split(".")[0] + "." if "." in response else response  # End at the first sentence

        # Add chatbot response to conversation
        st.session_state.conversation.append({"role": "chatbot", "text": response})
    else:
        st.warning("Please enter some text to start the conversation.")

# Display conversation history in a chat-like interface
st.subheader("Conversation History:")
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                <div style="background: #0078D4; color: white; padding: 10px; border-radius: 10px; max-width: 70%;">
                    {msg["text"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                <div style="background: #E1E1E1; color: black; padding: 10px; border-radius: 10px; max-width: 70%;">
                    {msg["text"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
