import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # 한 번에 병렬로 처리할 시퀀스의 수
block_size = 128 # 예측을 위한 최대 컨텍스트 길이
max_iters = 1000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

torch.manual_seed(1337)

class DataHandler:
    def __init__(self, data_path, block_size, batch_size, device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # 1. 토큰 임베딩 테이블: 각 문자를 벡터로 변환
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. 포지션 임베딩 테이블: 위치 정보를 벡터로 변환
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 3. 트랜스포머 블록들
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # 4. 최종 Layer Normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # 5. 최종 선형 계층 (Logits 생성)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss(model, data_handler):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_handler.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, data_handler):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data_handler)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = data_handler.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("학습 완료. (Training finished.)")

def main():
    print(f"Using device: {device}")
    
    # 1. 데이터 준비
    data_handler = DataHandler('data/tiny-shakespeare.txt', block_size, batch_size, device)
    print(f"Vocab size: {data_handler.vocab_size}")
    
    # 2. 모델 초기화
    model = GPTLanguageModel(data_handler.vocab_size)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # 3. 학습 수행
    train_model(m, data_handler)
    
    # 4. 생성 (Generation)
    print("생성된 텍스트 샘플 (Generated sample):")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = m.generate(context, max_new_tokens=500)[0].tolist()
    print(generated_indices)
    print(data_handler.decode(generated_indices))

if __name__ == '__main__':
    main()
