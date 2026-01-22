
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import LGConv
from torch_geometric.utils import structured_negative_sampling
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = 'data/ml-latest-small'
RATINGS_FILE = f'{DATA_PATH}/ratings.csv'
MOVIES_FILE = f'{DATA_PATH}/movies.csv'
BATCH_SIZE = 8192
EMBED_DIM = 64
NUM_LAYERS = 3
LR = 0.01
EPOCHS = 30
K = 20  # For Recall@K and NDCG@K
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --- Data Preprocessing ---
def load_data():
    print("Loading data...")
    ratings = pd.read_csv(RATINGS_FILE)
    
    # Implicit feedback: keep only good ratings (>= 3.5)
    ratings = ratings[ratings['rating'] >= 3.5]
    
    # Remap IDs to [0, N-1]
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}
    
    ratings['user_idx'] = ratings['userId'].map(user_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_map)
    
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    
    print(f"Num Users: {num_users}, Num Movies: {num_movies}, Num Interactions: {len(ratings)}")
    
    return ratings, num_users, num_movies, user_map, movie_map

def start_id_mapping(ratings, num_users, num_movies):
    # User -> Item edges
    edge_index_user_to_item = torch.tensor([
        ratings['user_idx'].values,
        ratings['movie_idx'].values
    ], dtype=torch.long) # [2, num_edges]

    train_edges, test_edges = train_test_split(
        edge_index_user_to_item.T, test_size=0.2, random_state=SEED
    )
    
    return train_edges.T, test_edges.T

# --- Model ---
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        emb = self.embedding.weight
        embs = [emb]
        
        for conv in self.convs:
            emb = conv(emb, edge_index)
            embs.append(emb)
            
        embs = torch.stack(embs, dim=1)
        embs_mean = torch.mean(embs, dim=1)
        
        return embs_mean

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        all_embs = self(edge_index)
        
        user_embs = all_embs[users]
        pos_item_embs = all_embs[self.num_users + pos_items]
        neg_item_embs = all_embs[self.num_users + neg_items]
        
        return user_embs, pos_item_embs, neg_item_embs
    
    def get_loss(self, users, pos_items, neg_items, edge_index):
        user_embs, pos_item_embs, neg_item_embs = self.encode_minibatch(users, pos_items, neg_items, edge_index)
        
        # BPR Loss
        pos_scores = torch.mul(user_embs, pos_item_embs).sum(dim=1)
        neg_scores = torch.mul(user_embs, neg_item_embs).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
        
        return loss

# --- Evaluation ---
def evaluate(model, train_edges, test_edges, k=20):
    pass 

# --- Main ---
def main():
    ratings, num_users, num_movies, user_map, movie_map = load_data()
    train_edge_index, test_edge_index = start_id_mapping(ratings, num_users, num_movies)
    
    train_edge_index = train_edge_index.to(DEVICE)
    test_edge_index = test_edge_index.to(DEVICE)
    
    def make_bipartite_graph(edge_index):
        users = edge_index[0]
        items = edge_index[1] + num_users 
        
        row = torch.cat([users, items])
        col = torch.cat([items, users])
        return torch.stack([row, col], dim=0)

    train_graph_edge_index = make_bipartite_graph(train_edge_index)
    
    model = LightGCN(num_users, num_movies, EMBED_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        perm = torch.randperm(train_edge_index.size(1))
        num_batches = (train_edge_index.size(1) + BATCH_SIZE - 1) // BATCH_SIZE
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i in pbar:
            batch_indices = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_users = train_edge_index[0, batch_indices]
            batch_pos_items = train_edge_index[1, batch_indices]
            
            batch_neg_items = torch.randint(0, num_movies, (len(batch_indices),), device=DEVICE)
            
            optimizer.zero_grad()
            loss = model.get_loss(batch_users, batch_pos_items, batch_neg_items, train_graph_edge_index)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    # Evaluation
    print("Evaluating...")
    model.eval()
    
    with torch.no_grad():
        all_embs = model(train_graph_edge_index)
        user_embs = all_embs[:num_users]
        item_embs = all_embs[num_users:]
        
        scores = torch.matmul(user_embs, item_embs.t())
        
        rows = train_edge_index[0].cpu().numpy()
        cols = train_edge_index[1].cpu().numpy()
        scores[rows, cols] = -float('inf')
        
        _, topk_indices = torch.topk(scores, K, dim=1)
        topk_indices = topk_indices.cpu().numpy()
        
        test_user_items = {}
        test_u = test_edge_index[0].cpu().numpy()
        test_i = test_edge_index[1].cpu().numpy()
        
        for u, i in zip(test_u, test_i):
            if u not in test_user_items:
                test_user_items[u] = set()
            test_user_items[u].add(i)
            
        recall_sum = 0
        ndcg_sum = 0
        num_test_users = 0
        
        for u, items_gt in test_user_items.items():
            if u >= num_users: continue
            num_test_users += 1
            
            rec_items = topk_indices[u]
            
            hits = 0
            dcg = 0
            idcg = 0
            
            for rank, item in enumerate(rec_items):
                if item in items_gt:
                    hits += 1
                    dcg += 1 / np.log2(rank + 2)
            
            num_gt = len(items_gt)
            for i in range(min(K, num_gt)):
                idcg += 1 / np.log2(i + 2)
            
            recall_sum += hits / num_gt if num_gt > 0 else 0
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_sum += ndcg
            
        avg_recall = recall_sum / num_test_users
        avg_ndcg = ndcg_sum / num_test_users
        
        print(f"Test Results -- Recall@{K}: {avg_recall:.4f}, NDCG@{K}: {avg_ndcg:.4f}")

if __name__ == "__main__":
    main()
