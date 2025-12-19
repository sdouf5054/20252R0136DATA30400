"""
GNN-based Classification with Hierarchical Structure
- Build class graph from hierarchy
- Use GCN to propagate information
- Simple implementation for 2-day deadline
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings('ignore')

# ==================== Paths ====================
PROJECT_ROOT = Path("/home/sagemaker-user/project_release/Amazon_products")

# Data files
TRAIN_CORPUS_PATH = PROJECT_ROOT / "train/train_corpus.txt"
TEST_CORPUS_PATH = PROJECT_ROOT / "test/test_corpus.txt"
SILVER_LABELS_PATH = PROJECT_ROOT / "outputs/silver_labels.json"
CLASSES_PATH = PROJECT_ROOT / "classes.txt"
HIERARCHY_PATH = PROJECT_ROOT / "class_hierarchy.txt"
KEYWORDS_PATH = PROJECT_ROOT / "class_related_keywords.txt"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission_gnn_3.csv"

# ==================== Parameters ====================
NUM_CLASSES = 531
TOP_K_PREDICT = 3
RANDOM_SEED = 42
MAX_FEATURES = 10000

# GNN parameters
GNN_HIDDEN_DIM = 128       # Reduced from 256
GNN_NUM_LAYERS = 2
NUM_EPOCHS = 10            # Increased to 10 for better convergence
BATCH_SIZE = 256           # Increased for speed
LEARNING_RATE = 0.001

# Hierarchy parameters
HIERARCHY_DEPTH = 1        # Add only 1 level of parents (None = all ancestors)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Load Data ====================

def load_corpus(path):
    """Load product_id -> text mapping"""
    pid2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                pid, text = parts
                pid2text[pid] = text
    return pid2text

def load_classes(path):
    """Load class_id -> class_name mapping"""
    id2class = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                class_id, class_name = int(parts[0]), parts[1]
                id2class[class_id] = class_name
    return id2class

def load_hierarchy(path):
    """Load parent -> children mapping and edge list (supports DAG)"""
    edges = []
    parent2children = defaultdict(list)
    child2parents = defaultdict(list)  # Support multiple parents
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                edges.append((parent, child))
                edges.append((child, parent))  # Undirected
                parent2children[parent].append(child)
                child2parents[child].append(parent)
    
    return edges, parent2children, child2parents

def load_keywords(path):
    """Load class_name -> keywords mapping"""
    class2keywords = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                class_name, keywords_str = line.strip().split(':', 1)
                keywords = keywords_str.split(',')
                class2keywords[class_name] = ' '.join(keywords)
    return class2keywords

def add_parent_labels(labels, child2parents, max_depth=None):
    """
    Add parent labels based on hierarchy (supports multiple parents via BFS)
    
    Args:
        labels: list of class IDs
        child2parents: dict mapping child to list of parents
        max_depth: maximum levels to traverse (None = all ancestors)
    
    Returns:
        sorted list of labels with parents added
    """
    final_labels = set(labels)
    queue = [(label, 0) for label in labels]  # (node, depth)
    visited = set(labels)
    
    while queue:
        current, depth = queue.pop(0)
        
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            continue
        
        if current in child2parents:
            for parent in child2parents[current]:
                if parent not in visited:
                    final_labels.add(parent)
                    visited.add(parent)
                    queue.append((parent, depth + 1))
    
    return sorted(list(final_labels))

# ==================== Build Graph ====================

def build_adjacency_matrix(edges, num_nodes):
    """
    Build normalized adjacency matrix for GCN
    A_norm = D^(-1/2) @ A @ D^(-1/2)
    """
    # Create adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for src, dst in edges:
        if src < num_nodes and dst < num_nodes:
            adj[src, dst] = 1.0
    
    # Add self-loops
    adj = adj + np.eye(num_nodes)
    
    # Degree matrix
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    
    # Normalized adjacency
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return torch.FloatTensor(adj_norm)

# ==================== GNN Model ====================

class GCNLayer(nn.Module):
    """Simple GCN layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x: (num_nodes, in_features)
            adj: (num_nodes, num_nodes) normalized adjacency
        """
        x = self.linear(x)
        x = torch.mm(adj, x)  # Graph convolution
        return x

class LabelGCN(nn.Module):
    """GCN for enriching label embeddings"""
    def __init__(self, num_classes, label_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, label_dim))
        
        # GCN layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(label_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, adj):
        """
        Args:
            adj: (num_classes, num_classes) normalized adjacency
        Returns:
            enriched_embeddings: (num_classes, hidden_dim)
        """
        x = self.label_embeddings
        
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x

class GNNClassifier(nn.Module):
    """Complete GNN-based classifier"""
    def __init__(self, input_dim, num_classes, label_dim, gnn_hidden_dim, gnn_num_layers):
        super().__init__()
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, gnn_hidden_dim)
        )
        
        # Label GCN
        self.label_gcn = LabelGCN(num_classes, label_dim, gnn_hidden_dim, gnn_num_layers)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch_size, input_dim) text features
            adj: (num_classes, num_classes) label graph
        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode text
        text_features = self.text_encoder(x)  # (batch_size, gnn_hidden_dim)
        
        # Get enriched label embeddings
        label_features = self.label_gcn(adj)  # (num_classes, gnn_hidden_dim)
        
        # Compute similarity (inner product)
        logits = torch.mm(text_features, label_features.t())  # (batch_size, num_classes)
        
        return logits

# ==================== Dataset ====================

# ==================== Dataset ====================

def sparse_collate_fn(batch):
    """
    Custom collate function to convert sparse to dense at batch level
    Much faster than per-sample conversion
    """
    if isinstance(batch[0], tuple):
        # Training: (X, y)
        indices = [item[0] for item in batch]
        ys = torch.stack([item[1] for item in batch])
        # Stack sparse vectors efficiently
        X_batch = torch.FloatTensor(np.vstack([idx.toarray() for idx in indices]))
        return X_batch, ys
    else:
        # Testing: X only
        X_batch = torch.FloatTensor(np.vstack([item.toarray() for item in batch]))
        return X_batch

class TextDataset(Dataset):
    def __init__(self, X, y=None):
        # Keep sparse format - convert in collate_fn
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Return sparse matrix (will be converted in collate_fn)
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        return x

# ==================== Training ====================

def train_epoch(model, dataloader, optimizer, criterion, adj, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch, adj)
        loss = criterion(logits, y_batch.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ==================== Prediction ====================

def predict_multi_label(model, dataloader, adj, device, top_k=3, child2parents=None, hierarchy_depth=None):
    """Predict top-k labels for each sample"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both tuple and tensor inputs safely
            X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            X_batch = X_batch.to(device)
            
            logits = model(X_batch, adj)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            for score in scores:
                # Get top-k
                top_k_indices = np.argsort(score)[-top_k:][::-1]
                labels = top_k_indices.tolist()
                
                # Add parent labels with depth control
                if child2parents:
                    labels = add_parent_labels(labels, child2parents, max_depth=hierarchy_depth)
                
                all_predictions.append(labels)
    
    return all_predictions

# ==================== Main ====================

def main():
    print("="*60)
    print("GNN-BASED HIERARCHICAL CLASSIFICATION")
    print("="*60)
    
    # Load data
    print("\n[1/8] Loading data...")
    train_pid2text = load_corpus(TRAIN_CORPUS_PATH)
    test_pid2text = load_corpus(TEST_CORPUS_PATH)
    
    train_pids = list(train_pid2text.keys())
    test_pids = list(test_pid2text.keys())
    train_texts = [train_pid2text[pid] for pid in train_pids]
    test_texts = [test_pid2text[pid] for pid in test_pids]
    
    print(f"  ✓ Train: {len(train_pids):,} samples")
    print(f"  ✓ Test: {len(test_pids):,} samples")
    
    # Load metadata
    print("\n[2/8] Loading metadata...")
    id2class = load_classes(CLASSES_PATH)
    edges, parent2children, child2parents = load_hierarchy(HIERARCHY_PATH)
    class2keywords = load_keywords(KEYWORDS_PATH)
    
    with open(SILVER_LABELS_PATH, 'r') as f:
        silver_labels = json.load(f)
    
    print(f"  ✓ Classes: {len(id2class)}")
    print(f"  ✓ Hierarchy edges: {len(edges)}")
    print(f"  ✓ Silver labels: {len(silver_labels):,}")
    
    # Build graph adjacency matrix
    print("\n[3/8] Building label graph...")
    adj_matrix = build_adjacency_matrix(edges, NUM_CLASSES).to(device)
    print(f"  ✓ Adjacency matrix: {adj_matrix.shape}")
    
    # TF-IDF vectorization
    print("\n[4/8] Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train_all = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Prepare training data
    train_pid2idx = {pid: idx for idx, pid in enumerate(train_pids)}  # O(1) lookup
    labeled_pids = [pid for pid in train_pids if pid in silver_labels]
    labeled_indices = [train_pid2idx[pid] for pid in labeled_pids]
    X_train = X_train_all[labeled_indices]
    labeled_labels = [silver_labels[pid] for pid in labeled_pids]
    
    mlb = MultiLabelBinarizer(classes=range(NUM_CLASSES))
    y_train = mlb.fit_transform(labeled_labels)
    
    # Filter valid classes (relaxed for GNN)
    class_counts = y_train.sum(axis=0)
    valid_classes = class_counts >= 1  # Changed from 2 to 1 (more classes for GNN)
    y_train_filtered = y_train[:, valid_classes]
    valid_class_indices = np.where(valid_classes)[0]
    
    print(f"  ✓ Valid classes: {len(valid_class_indices)} / {NUM_CLASSES}")
    print(f"  ✓ Training samples: {X_train.shape[0]:,}")
    
    # Create datasets
    train_dataset = TextDataset(X_train, torch.FloatTensor(y_train_filtered))
    test_dataset = TextDataset(X_test)
    
    # Use custom collate_fn for efficient sparse->dense conversion
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=sparse_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            collate_fn=sparse_collate_fn)
    
    # Initialize model
    print("\n[5/8] Initializing GNN model...")
    
    # Initialize label embeddings with keywords (deterministic)
    label_dim = 128
    label_init_embeddings = np.zeros((NUM_CLASSES, label_dim), dtype=np.float32)
    
    # Use deterministic hash for each class
    for class_id, class_name in id2class.items():
        if class_name in class2keywords:
            keyword_str = class2keywords[class_name]
            # Use deterministic seed based on class_id (not hash)
            seed = (RANDOM_SEED + class_id) % (2**32)
            rng = np.random.RandomState(seed)
            label_init_embeddings[class_id] = rng.randn(label_dim).astype(np.float32)
        else:
            # For classes without keywords, use class_id as seed
            seed = (RANDOM_SEED + class_id + 1000) % (2**32)
            rng = np.random.RandomState(seed)
            label_init_embeddings[class_id] = rng.randn(label_dim).astype(np.float32)
    
    # Normalize embeddings
    label_init_embeddings = label_init_embeddings / (np.linalg.norm(label_init_embeddings, axis=1, keepdims=True) + 1e-8)
    
    model = GNNClassifier(
        input_dim=MAX_FEATURES,
        num_classes=len(valid_class_indices),
        label_dim=label_dim,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_num_layers=GNN_NUM_LAYERS
    ).to(device)
    
    # ✅ FIX: Actually inject the keyword-based embeddings into model
    with torch.no_grad():
        filtered_label_embeddings = label_init_embeddings[valid_class_indices]
        model.label_gcn.label_embeddings.data = torch.from_numpy(filtered_label_embeddings).to(device)
    
    # Filter adjacency matrix for valid classes
    adj_filtered = adj_matrix[valid_class_indices][:, valid_class_indices]
    
    print(f"  ✓ Model initialized")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n[6/8] Training GNN model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, adj_filtered, device)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")
    
    print("  ✓ Training complete!")
    
    # Prediction
    print("\n[7/8] Predicting on test set...")
    test_predictions_filtered = predict_multi_label(
        model, test_loader, adj_filtered, device, 
        top_k=TOP_K_PREDICT, child2parents=None, hierarchy_depth=None  # No hierarchy in filtered space
    )
    
    # Map back to original class indices and add parents (with depth control)
    test_predictions = []
    for pred_filtered in test_predictions_filtered:
        pred_original = [valid_class_indices[idx] for idx in pred_filtered if idx < len(valid_class_indices)]
        # Add parents in original space with HIERARCHY_DEPTH
        pred_with_parents = add_parent_labels(pred_original, child2parents, max_depth=HIERARCHY_DEPTH)
        test_predictions.append(pred_with_parents)
    
    # Generate submission
    print("\n[8/8] Generating submission...")
    submission_data = []
    for pid, labels in zip(test_pids, test_predictions):
        labels_str = ",".join(map(str, labels))
        submission_data.append({"id": pid, "label": labels_str})
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"  ✓ Submission saved to: {SUBMISSION_PATH}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model type:           GNN (LabelGCN)")
    print(f"GNN layers:           {GNN_NUM_LAYERS}")
    print(f"Hidden dim:           {GNN_HIDDEN_DIM}")
    print(f"Hierarchy depth:      {HIERARCHY_DEPTH if HIERARCHY_DEPTH else 'all ancestors'}")
    print(f"Training samples:     {len(labeled_pids):,}")
    print(f"Test predictions:     {len(test_pids):,}")
    print(f"Avg labels per pred:  {np.mean([len(p) for p in test_predictions]):.2f}")
    print("\n" + "="*60)
    print("✅ DONE! Ready for Kaggle submission")
    print("="*60)

if __name__ == "__main__":
    main()
