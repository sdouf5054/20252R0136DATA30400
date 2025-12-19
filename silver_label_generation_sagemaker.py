"""
Silver Label Generation for Project (SageMaker Version)
- Keyword matching based approach
- Hierarchy-aware label propagation
- Noise control with top-k selection
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ==================== Paths (SageMaker) ====================
PROJECT_ROOT = Path("/home/sagemaker-user/project_release/Amazon_products")
CLASSES_PATH = PROJECT_ROOT / "classes.txt"
HIERARCHY_PATH = PROJECT_ROOT / "class_hierarchy.txt"
KEYWORDS_PATH = PROJECT_ROOT / "class_related_keywords.txt"

# Training corpus
TRAIN_CORPUS_PATH = PROJECT_ROOT / "train/train_corpus.txt"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "silver_labels.json"

# ==================== Parameters ====================
TOP_K = 3  # Maximum labels per sample (Samples F1 optimal)
MIN_KEYWORD_MATCHES = 1  # Minimum keyword matches to consider

# ==================== Load Data ====================

def load_classes(path):
    """Load class_id -> class_name mapping"""
    id2class = {}
    class2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                class_id, class_name = int(parts[0]), parts[1]
                id2class[class_id] = class_name
                class2id[class_name] = class_id
    return id2class, class2id

def load_hierarchy(path):
    """Load parent -> children mapping"""
    parent2children = defaultdict(list)
    child2parent = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                parent2children[parent].append(child)
                child2parent[child] = parent
    return parent2children, child2parent

def load_keywords(path):
    """Load class_name -> keywords mapping"""
    class2keywords = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                class_name, keywords_str = line.strip().split(':', 1)
                keywords = keywords_str.split(',')
                class2keywords[class_name] = keywords
    return class2keywords

def load_corpus(path):
    """Load product_id -> text mapping"""
    pid2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                pid, text = parts
                pid2text[pid] = text.lower()  # Lowercase for matching
    return pid2text

# ==================== Silver Label Generation ====================

def match_keywords(text, keywords):
    """Count how many keywords appear in text"""
    text_lower = text.lower()
    matches = []
    for kw in keywords:
        kw_clean = kw.strip().replace('_', ' ')
        if kw_clean in text_lower:
            matches.append(kw_clean)
    return matches

def get_all_ancestors(class_id, child2parent):
    """Get all ancestor classes in hierarchy"""
    ancestors = []
    current = class_id
    while current in child2parent:
        parent = child2parent[current]
        ancestors.append(parent)
        current = parent
    return ancestors

def generate_silver_labels(pid2text, class2keywords, class2id, child2parent, 
                          top_k=3, min_matches=1):
    """
    Generate silver labels using keyword matching
    
    Returns:
        pid2labels: dict mapping product_id -> list of class_ids
        stats: dict with statistics
    """
    pid2labels = {}
    stats = {
        'total_samples': len(pid2text),
        'labeled_samples': 0,
        'discarded_samples': 0,
        'avg_labels_per_sample': 0,
        'label_distribution': defaultdict(int)
    }
    
    for pid, text in tqdm(pid2text.items(), desc="Generating silver labels"):
        # Score each class
        class_scores = {}
        
        for class_name, keywords in class2keywords.items():
            matches = match_keywords(text, keywords)
            if len(matches) >= min_matches:
                class_id = class2id[class_name]
                class_scores[class_id] = len(matches)
        
        # Select top-k classes
        if not class_scores:
            stats['discarded_samples'] += 1
            continue
        
        # Sort by score and take top-k
        top_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected_labels = set([cls_id for cls_id, _ in top_classes])
        
        # Add hierarchy: if child is selected, include all parents
        final_labels = set(selected_labels)
        for cls_id in selected_labels:
            ancestors = get_all_ancestors(cls_id, child2parent)
            final_labels.update(ancestors)
        
        final_labels = sorted(list(final_labels))
        pid2labels[pid] = final_labels
        
        stats['labeled_samples'] += 1
        for label in final_labels:
            stats['label_distribution'][label] += 1
    
    # Calculate average
    if stats['labeled_samples'] > 0:
        total_labels = sum(len(labels) for labels in pid2labels.values())
        stats['avg_labels_per_sample'] = total_labels / stats['labeled_samples']
    
    return pid2labels, stats

# ==================== Main ====================

def main():
    print("="*60)
    print("SILVER LABEL GENERATION")
    print("="*60)
    
    print("\n[1/5] Loading metadata...")
    id2class, class2id = load_classes(CLASSES_PATH)
    parent2children, child2parent = load_hierarchy(HIERARCHY_PATH)
    class2keywords = load_keywords(KEYWORDS_PATH)
    
    print(f"  ✓ Loaded {len(id2class)} classes")
    print(f"  ✓ Loaded {len(class2keywords)} keyword sets")
    print(f"  ✓ Loaded {len(child2parent)} hierarchy edges")
    
    print("\n[2/5] Loading training corpus...")
    if not TRAIN_CORPUS_PATH.exists():
        print(f"  ❌ Training corpus not found at: {TRAIN_CORPUS_PATH}")
        return
    
    pid2text = load_corpus(TRAIN_CORPUS_PATH)
    print(f"  ✓ Loaded {len(pid2text)} training samples")
    
    print("\n[3/5] Generating silver labels...")
    print(f"  • TOP_K = {TOP_K}")
    print(f"  • MIN_KEYWORD_MATCHES = {MIN_KEYWORD_MATCHES}")
    
    pid2labels, stats = generate_silver_labels(
        pid2text, class2keywords, class2id, child2parent,
        top_k=TOP_K, min_matches=MIN_KEYWORD_MATCHES
    )
    
    print("\n[4/5] Saving results...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(pid2labels, f, indent=2)
    print(f"  ✓ Silver labels saved to: {OUTPUT_PATH}")
    
    # Save stats
    stats_path = OUTPUT_DIR / "silver_labels_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convert defaultdict to dict for JSON serialization
        stats_serializable = dict(stats)
        stats_serializable['label_distribution'] = dict(stats['label_distribution'])
        json.dump(stats_serializable, f, indent=2)
    print(f"  ✓ Statistics saved to: {stats_path}")
    
    print("\n[5/5] Summary")
    print("="*60)
    print(f"Total samples:        {stats['total_samples']:,}")
    print(f"Labeled samples:      {stats['labeled_samples']:,}")
    print(f"Discarded samples:    {stats['discarded_samples']:,}")
    print(f"Avg labels/sample:    {stats['avg_labels_per_sample']:.2f}")
    
    print("\nTop 10 most frequent labels:")
    print("-" * 60)
    top_labels = sorted(stats['label_distribution'].items(), 
                       key=lambda x: x[1], reverse=True)[:10]
    for label_id, count in top_labels:
        class_name = id2class[label_id]
        print(f"  {label_id:3d}  {class_name:35s}  {count:6,}")
    
    print("\n" + "="*60)
    print("✅ DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
