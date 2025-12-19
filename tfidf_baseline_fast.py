"""
TF-IDF Baseline Model for Project (FAST VERSION)
- TF-IDF vectorization (sklearn optimized)
- Train classifier using silver labels
- Generate Kaggle submission
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# ==================== Paths ====================
PROJECT_ROOT = Path("/home/sagemaker-user/project_release/Amazon_products")

# Data files
TRAIN_CORPUS_PATH = PROJECT_ROOT / "train/train_corpus.txt"
TEST_CORPUS_PATH = PROJECT_ROOT / "test/test_corpus.txt"
SILVER_LABELS_PATH = PROJECT_ROOT / "outputs/silver_labels.json"
HIERARCHY_PATH = PROJECT_ROOT / "class_hierarchy.txt"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission_tfidf_baseline.csv"

# ==================== Parameters ====================
NUM_CLASSES = 531
TOP_K_PREDICT = 3  # Predict top-k labels for Samples F1
RANDOM_SEED = 42
MAX_FEATURES = 10000  # Vocabulary size limit for speed

np.random.seed(RANDOM_SEED)

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

def load_hierarchy(path):
    """Load hierarchy for post-processing"""
    child2parent = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                child2parent[child] = parent
    return child2parent

# ==================== Hierarchy Post-processing ====================

def add_parent_labels(labels, child2parent):
    """Add all parent labels based on hierarchy"""
    final_labels = set(labels)
    for label in labels:
        current = label
        while current in child2parent:
            parent = child2parent[current]
            final_labels.add(parent)
            current = parent
    return sorted(list(final_labels))

# ==================== Training ====================

def train_model(X_train, y_train):
    """Train multi-label classifier"""
    print("\n[Training] Training classifier...")
    print(f"  • Input dim: {X_train.shape[1]}")
    print(f"  • Num classes: {y_train.shape[1]}")
    print(f"  • Training samples: {X_train.shape[0]}")
    
    # Use LogisticRegression with OneVsRest approach
    base_clf = LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    
    clf = MultiOutputClassifier(base_clf, n_jobs=-1)
    
    print("  → Training... (this may take 5-10 minutes)")
    clf.fit(X_train, y_train)
    
    print("  ✓ Training complete!")
    return clf

# ==================== Prediction ====================

def predict_top_k(clf, X, k=3, child2parent=None, valid_class_indices=None):
    """
    Predict top-k labels with hierarchy post-processing
    
    Returns:
        predictions: list of label lists
    """
    print(f"\n[Prediction] Predicting top-{k} labels...")
    
    # Get decision function scores (faster than predict_proba)
    print("  → Computing scores...")
    y_scores = []
    for i, estimator in enumerate(clf.estimators_):
        scores = estimator.decision_function(X)
        y_scores.append(scores)
    
    y_scores = np.array(y_scores).T  # (n_samples, n_valid_classes)
    
    print("  → Selecting top-k...")
    predictions = []
    n_samples = X.shape[0]  # Fix: use shape[0] for sparse matrix
    for i in tqdm(range(n_samples), desc="Post-processing"):
        # Get top-k classes (map back to original indices)
        top_k_valid = np.argsort(y_scores[i])[-k:][::-1]
        top_labels = [valid_class_indices[idx] for idx in top_k_valid]
        
        # Add parent labels
        if child2parent:
            top_labels = add_parent_labels(top_labels, child2parent)
        
        predictions.append(top_labels)
    
    print(f"  ✓ Predictions complete!")
    return predictions

# ==================== Main ====================

def main():
    print("="*60)
    print("TF-IDF BASELINE MODEL (FAST)")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading corpus...")
    train_pid2text = load_corpus(TRAIN_CORPUS_PATH)
    test_pid2text = load_corpus(TEST_CORPUS_PATH)
    
    train_pids = list(train_pid2text.keys())
    test_pids = list(test_pid2text.keys())
    train_texts = [train_pid2text[pid] for pid in train_pids]
    test_texts = [test_pid2text[pid] for pid in test_pids]
    
    print(f"  ✓ Train: {len(train_pids):,} samples")
    print(f"  ✓ Test: {len(test_pids):,} samples")
    
    # Load silver labels
    print("\n[2/6] Loading silver labels...")
    with open(SILVER_LABELS_PATH, 'r') as f:
        silver_labels = json.load(f)
    
    print(f"  ✓ Loaded silver labels for {len(silver_labels):,} samples")
    
    # Load hierarchy
    print("\n[3/6] Loading hierarchy...")
    child2parent = load_hierarchy(HIERARCHY_PATH)
    print(f"  ✓ Loaded {len(child2parent)} hierarchy edges")
    
    # Filter train data to only labeled samples
    labeled_pids = [pid for pid in train_pids if pid in silver_labels]
    labeled_texts = [train_pid2text[pid] for pid in labeled_pids]
    labeled_labels = [silver_labels[pid] for pid in labeled_pids]
    
    print(f"\n  → Using {len(labeled_pids):,} / {len(train_pids):,} labeled samples")
    
    # TF-IDF Vectorization
    print("\n[4/6] Creating TF-IDF vectors...")
    print(f"  • Max features: {MAX_FEATURES:,}")
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),  # unigrams + bigrams
        lowercase=True,
        strip_accents='unicode'
    )
    
    print("  → Fitting on all train texts...")
    X_all_train = vectorizer.fit_transform(train_texts)
    
    print("  → Transforming test texts...")
    X_test = vectorizer.transform(test_texts)
    
    # Extract vectors for labeled samples only
    labeled_indices = [train_pids.index(pid) for pid in labeled_pids]
    X_train = X_all_train[labeled_indices]
    
    # Prepare labels
    mlb = MultiLabelBinarizer(classes=range(NUM_CLASSES))
    y_train = mlb.fit_transform(labeled_labels)
    
    # Filter out classes with too few positive samples
    print("\n  → Filtering classes...")
    class_counts = y_train.sum(axis=0)
    valid_classes = class_counts >= 2  # At least 2 positive samples
    n_valid = valid_classes.sum()
    n_removed = NUM_CLASSES - n_valid
    
    print(f"  • Valid classes: {n_valid} / {NUM_CLASSES}")
    print(f"  • Removed classes: {n_removed} (< 2 samples)")
    
    # Keep only valid classes
    y_train = y_train[:, valid_classes]
    valid_class_indices = np.where(valid_classes)[0]
    
    print(f"\n  ✓ X_train: {X_train.shape}")
    print(f"  ✓ y_train: {y_train.shape}")
    print(f"  ✓ X_test: {X_test.shape}")
    
    # Train model
    print("\n[5/6] Training model...")
    clf = train_model(X_train, y_train)
    
    # Predict on test
    print("\n[6/6] Predicting on test set...")
    test_predictions = predict_top_k(clf, X_test, k=TOP_K_PREDICT, 
                                    child2parent=child2parent,
                                    valid_class_indices=valid_class_indices)
    
    # Generate submission
    print("\n[7/7] Generating submission file...")
    submission_data = []
    for pid, labels in zip(test_pids, test_predictions):
        labels_str = ",".join(map(str, labels))
        submission_data.append({"id": pid, "label": labels_str})
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"  ✓ Submission saved to: {SUBMISSION_PATH}")
    
    # Statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples:     {len(labeled_pids):,}")
    print(f"Test samples:         {len(test_pids):,}")
    print(f"Vocabulary size:      {X_train.shape[1]:,}")
    print(f"Avg labels per pred:  {np.mean([len(p) for p in test_predictions]):.2f}")
    print("\nSubmission preview:")
    print(submission_df.head(10))
    print("\n" + "="*60)
    print("✅ DONE! Ready for Kaggle submission")
    print("="*60)

if __name__ == "__main__":
    main()
