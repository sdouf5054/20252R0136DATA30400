"""
Self-Training with Pseudo-Labeling
- Use trained model to predict unlabeled data
- Select high-confidence predictions as pseudo-labels
- Retrain with combined silver + pseudo labels
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
PSEUDO_LABELS_PATH = OUTPUT_DIR / "pseudo_labels.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission_selftrain.csv"

# ==================== Parameters ====================
NUM_CLASSES = 531
TOP_K_PREDICT = 3
RANDOM_SEED = 42
MAX_FEATURES = 10000

# Self-training parameters
CONFIDENCE_THRESHOLD = 0.5     # Lowered from 0.9 to get more pseudo-labels
TOP_K_PSEUDO = 1                # Number of labels for pseudo-labeling (conservative)
MIN_PSEUDO_LABELS = 100         # Lowered from 1000 (more realistic)
PSEUDO_HIERARCHY_DEPTH = 1      # Add only 1 level of parents for pseudo-labels (0=none, 1=parent only, -1=all ancestors)

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

def add_parent_labels(labels, child2parent, depth=-1):
    """
    Add parent labels based on hierarchy
    
    Args:
        labels: list of class IDs
        child2parent: dict mapping child to parent
        depth: how many levels to add
            -1 = all ancestors (original behavior)
            0 = no parents
            1 = only direct parent
            2 = parent + grandparent, etc.
    
    Returns:
        sorted list of labels with parents added
    """
    if depth == 0:
        return sorted(list(set(labels)))
    
    final_labels = set(labels)
    for label in labels:
        current = label
        levels = 0
        while current in child2parent and (depth == -1 or levels < depth):
            parent = child2parent[current]
            final_labels.add(parent)
            current = parent
            levels += 1
    return sorted(list(final_labels))

# ==================== Training ====================

def train_model(X_train, y_train, verbose=True):
    """Train multi-label classifier"""
    if verbose:
        print("\n[Training] Training classifier...")
        print(f"  • Input dim: {X_train.shape[1]}")
        print(f"  • Num classes: {y_train.shape[1]}")
        print(f"  • Training samples: {X_train.shape[0]}")
    
    base_clf = LogisticRegression(
        max_iter=100,
        solver='lbfgs',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    
    clf = MultiOutputClassifier(base_clf, n_jobs=-1)
    
    if verbose:
        print("  → Training...")
    clf.fit(X_train, y_train)
    
    if verbose:
        print("  ✓ Training complete!")
    return clf

# ==================== Pseudo-labeling ====================

def generate_pseudo_labels(clf, X_unlabeled, unlabeled_pids, valid_class_indices, 
                          confidence_threshold=0.9, top_k_pseudo=1, hierarchy_depth=1, 
                          child2parent=None):
    """
    Generate pseudo-labels for unlabeled data
    
    Args:
        top_k_pseudo: number of top labels to include (conservative for quality)
        hierarchy_depth: how many parent levels to add
    
    Returns:
        pseudo_labels: dict {pid: [label_ids]}
        stats: dict with statistics
    """
    print(f"\n[Pseudo-labeling] Generating pseudo-labels...")
    print(f"  • Confidence threshold: {confidence_threshold}")
    print(f"  • Top-K for pseudo: {top_k_pseudo}")
    print(f"  • Hierarchy depth: {hierarchy_depth}")
    print(f"  • Unlabeled samples: {len(unlabeled_pids):,}")
    
    # Get decision scores
    print("  → Computing scores...")
    y_scores = []
    for estimator in clf.estimators_:
        scores = estimator.decision_function(X_unlabeled)
        y_scores.append(scores)
    
    y_scores = np.array(y_scores).T  # (n_samples, n_valid_classes)
    
    # Apply sigmoid to get probabilities (approximate)
    y_proba = 1 / (1 + np.exp(-y_scores))
    
    # Select high-confidence predictions
    print("  → Selecting high-confidence predictions...")
    pseudo_labels = {}
    confidence_scores = []
    
    for i in tqdm(range(len(unlabeled_pids)), desc="Processing"):
        pid = unlabeled_pids[i]
        
        # Get top-k predictions with confidence
        top_k_indices = np.argsort(y_proba[i])[-top_k_pseudo:][::-1]  # Use TOP_K_PSEUDO
        top_scores = y_proba[i][top_k_indices]
        
        # Check if max confidence exceeds threshold
        max_confidence = top_scores[0]
        if max_confidence >= confidence_threshold:
            # Map back to original class indices
            top_labels = [valid_class_indices[idx] for idx in top_k_indices]
            
            # Add parent labels with limited depth
            if child2parent and hierarchy_depth != 0:
                top_labels = add_parent_labels(top_labels, child2parent, depth=hierarchy_depth)
            
            pseudo_labels[pid] = top_labels
            confidence_scores.append(max_confidence)
    
    # Statistics
    stats = {
        'total_unlabeled': len(unlabeled_pids),
        'pseudo_labeled': len(pseudo_labels),
        'acceptance_rate': len(pseudo_labels) / len(unlabeled_pids) if unlabeled_pids else 0,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
        'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
        'avg_labels_per_sample': np.mean([len(labels) for labels in pseudo_labels.values()]) if pseudo_labels else 0
    }
    
    print(f"\n  ✓ Pseudo-labels generated!")
    print(f"    • Accepted: {stats['pseudo_labeled']:,} / {stats['total_unlabeled']:,} ({stats['acceptance_rate']:.1%})")
    print(f"    • Avg confidence: {stats['avg_confidence']:.3f}")
    print(f"    • Avg labels/sample: {stats['avg_labels_per_sample']:.2f}")
    
    return pseudo_labels, stats

# ==================== Prediction ====================

def predict_top_k(clf, X, k=3, child2parent=None, valid_class_indices=None, hierarchy_depth=-1):
    """
    Predict top-k labels with hierarchy post-processing
    
    Args:
        hierarchy_depth: -1 for all ancestors (final prediction), or limited depth
    """
    print(f"\n[Prediction] Predicting top-{k} labels...")
    
    # Get decision function scores
    print("  → Computing scores...")
    y_scores = []
    for estimator in clf.estimators_:
        scores = estimator.decision_function(X)
        y_scores.append(scores)
    
    y_scores = np.array(y_scores).T
    
    print("  → Selecting top-k...")
    predictions = []
    n_samples = X.shape[0]
    for i in tqdm(range(n_samples), desc="Post-processing"):
        top_k_valid = np.argsort(y_scores[i])[-k:][::-1]
        top_labels = [valid_class_indices[idx] for idx in top_k_valid]
        
        if child2parent:
            top_labels = add_parent_labels(top_labels, child2parent, depth=hierarchy_depth)
        
        predictions.append(top_labels)
    
    print(f"  ✓ Predictions complete!")
    return predictions

# ==================== Main ====================

def main():
    print("="*60)
    print("SELF-TRAINING WITH PSEUDO-LABELING")
    print("="*60)
    
    # Load data
    print("\n[1/9] Loading corpus...")
    train_pid2text = load_corpus(TRAIN_CORPUS_PATH)
    test_pid2text = load_corpus(TEST_CORPUS_PATH)
    
    train_pids = list(train_pid2text.keys())
    test_pids = list(test_pid2text.keys())
    train_texts = [train_pid2text[pid] for pid in train_pids]
    test_texts = [test_pid2text[pid] for pid in test_pids]
    
    # Create pid2idx for O(1) lookup (avoid O(n²) later)
    train_pid2idx = {pid: idx for idx, pid in enumerate(train_pids)}
    
    print(f"  ✓ Train: {len(train_pids):,} samples")
    print(f"  ✓ Test: {len(test_pids):,} samples")
    
    # Load silver labels
    print("\n[2/9] Loading silver labels...")
    with open(SILVER_LABELS_PATH, 'r') as f:
        silver_labels = json.load(f)
    print(f"  ✓ Loaded {len(silver_labels):,} silver labels")
    
    # Load hierarchy
    print("\n[3/9] Loading hierarchy...")
    child2parent = load_hierarchy(HIERARCHY_PATH)
    print(f"  ✓ Loaded {len(child2parent)} hierarchy edges")
    
    # Separate labeled and unlabeled
    labeled_pids = [pid for pid in train_pids if pid in silver_labels]
    unlabeled_pids = [pid for pid in train_pids if pid not in silver_labels]
    
    print(f"\n  → Labeled: {len(labeled_pids):,}")
    print(f"  → Unlabeled: {len(unlabeled_pids):,}")
    
    # TF-IDF Vectorization
    print("\n[4/9] Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )
    
    X_all_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Prepare initial training data (silver labels only)
    labeled_indices = [train_pid2idx[pid] for pid in labeled_pids]
    X_labeled = X_all_train[labeled_indices]
    labeled_labels = [silver_labels[pid] for pid in labeled_pids]
    
    mlb = MultiLabelBinarizer(classes=range(NUM_CLASSES))
    y_labeled = mlb.fit_transform(labeled_labels)
    
    # Filter classes
    class_counts = y_labeled.sum(axis=0)
    valid_classes = class_counts >= 2
    y_labeled = y_labeled[:, valid_classes]
    valid_class_indices = np.where(valid_classes)[0]
    
    print(f"  ✓ Valid classes: {len(valid_class_indices)} / {NUM_CLASSES}")
    
    # Train initial model
    print("\n[5/9] Training initial model (silver labels only)...")
    clf_initial = train_model(X_labeled, y_labeled)
    
    # Generate pseudo-labels for unlabeled data
    print("\n[6/9] Generating pseudo-labels...")
    unlabeled_indices = [train_pid2idx[pid] for pid in unlabeled_pids]
    X_unlabeled = X_all_train[unlabeled_indices]
    
    pseudo_labels, pseudo_stats = generate_pseudo_labels(
        clf_initial, X_unlabeled, unlabeled_pids, 
        valid_class_indices, CONFIDENCE_THRESHOLD, 
        top_k_pseudo=TOP_K_PSEUDO,
        hierarchy_depth=PSEUDO_HIERARCHY_DEPTH,
        child2parent=child2parent
    )
    
    # Save pseudo-labels (convert numpy types to Python types for JSON)
    with open(PSEUDO_LABELS_PATH, 'w') as f:
        pseudo_labels_serializable = {
            pid: [int(label) for label in labels]  # Convert numpy int64 to Python int
            for pid, labels in pseudo_labels.items()
        }
        stats_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in pseudo_stats.items()
        }
        json.dump({
            'pseudo_labels': pseudo_labels_serializable,
            'stats': stats_serializable
        }, f, indent=2)
    print(f"  ✓ Pseudo-labels saved to: {PSEUDO_LABELS_PATH}")
    
    # Check if we have enough pseudo-labels
    if len(pseudo_labels) < MIN_PSEUDO_LABELS:
        print(f"\n  ⚠ Warning: Only {len(pseudo_labels)} pseudo-labels (< {MIN_PSEUDO_LABELS})")
        print(f"  → Continuing with available pseudo-labels...")
    
    # Combine silver + pseudo labels
    print("\n[7/9] Combining silver + pseudo labels...")
    combined_pids = labeled_pids + list(pseudo_labels.keys())
    combined_labels = [silver_labels[pid] for pid in labeled_pids] + \
                     [pseudo_labels[pid] for pid in pseudo_labels.keys()]
    
    combined_indices = [train_pid2idx[pid] for pid in combined_pids]
    X_combined = X_all_train[combined_indices]
    y_combined_full = mlb.fit_transform(combined_labels)
    y_combined = y_combined_full[:, valid_classes]
    
    print(f"  ✓ Combined training set: {len(combined_pids):,} samples")
    print(f"    • Silver labels: {len(labeled_pids):,}")
    print(f"    • Pseudo labels: {len(pseudo_labels):,}")
    
    # Retrain with combined data
    print("\n[8/9] Retraining with combined data...")
    clf_final = train_model(X_combined, y_combined)
    
    # Predict on test (use full hierarchy for final predictions)
    print("\n[9/9] Predicting on test set...")
    test_predictions = predict_top_k(clf_final, X_test, k=TOP_K_PREDICT,
                                    child2parent=child2parent,
                                    valid_class_indices=valid_class_indices,
                                    hierarchy_depth=-1)  # Full hierarchy for final prediction
    
    # Generate submission
    print("\n[Final] Generating submission file...")
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
    print(f"Initial training:     {len(labeled_pids):,} samples")
    print(f"Pseudo-labels added:  {len(pseudo_labels):,} samples")
    print(f"Final training:       {len(combined_pids):,} samples")
    print(f"Improvement:          +{len(pseudo_labels) / len(labeled_pids) * 100:.1f}% data")
    print(f"\nTest predictions:     {len(test_pids):,} samples")
    print(f"Avg labels per pred:  {np.mean([len(p) for p in test_predictions]):.2f}")
    print("\n" + "="*60)
    print("✅ DONE! Ready for Kaggle submission")
    print("="*60)

if __name__ == "__main__":
    main()
