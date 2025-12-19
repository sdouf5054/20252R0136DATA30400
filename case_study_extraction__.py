"""
Case Study Extraction Tool
- Find success and failure examples for report
- Analyze predictions vs ground truth (if available) or keyword matches
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from collections import defaultdict

# ==================== Paths ====================
PROJECT_ROOT = Path("/home/sagemaker-user/project_release/Amazon_products")

# Input files
TEST_CORPUS_PATH = PROJECT_ROOT / "test/test_corpus.txt"
CLASSES_PATH = PROJECT_ROOT / "classes.txt"
KEYWORDS_PATH = PROJECT_ROOT / "class_related_keywords.txt"
HIERARCHY_PATH = PROJECT_ROOT / "class_hierarchy.txt"

# Submissions
TFIDF_SUBMISSION = PROJECT_ROOT / "outputs/submission_tfidf_baseline.csv"
GNN_SUBMISSION = PROJECT_ROOT / "outputs/submission_gnn.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CASE_STUDY_PATH = OUTPUT_DIR / "case_study_examples.json"

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

def load_keywords(path):
    """Load class_name -> keywords mapping"""
    class2keywords = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                class_name, keywords_str = line.strip().split(':', 1)
                keywords = [kw.strip() for kw in keywords_str.split(',')]
                class2keywords[class_name] = keywords
    return class2keywords

def load_hierarchy(path):
    """Load parent -> children mapping (supports DAG with multiple parents)"""
    child2parents = defaultdict(list)  # Changed to support multiple parents
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                child2parents[child].append(parent)
    return child2parents

def load_submission(path):
    """Load submission file"""
    df = pd.read_csv(path)
    pid2labels = {}
    for _, row in df.iterrows():
        pid = str(row['id'])
        labels = [int(x) for x in str(row['label']).split(',')]
        pid2labels[pid] = labels
    return pid2labels

# ==================== Analysis ====================

def count_keyword_matches(text, keywords):
    """Count how many keywords appear in text"""
    text_lower = text.lower()
    matches = []
    for kw in keywords:
        kw_clean = kw.strip().replace('_', ' ').lower()
        if kw_clean in text_lower:
            matches.append(kw_clean)
    return matches

def analyze_prediction(pid, text, predicted_labels, id2class, class2keywords):
    """Analyze why this prediction was made"""
    analysis = {
        'pid': pid,
        'text': text[:200] + '...' if len(text) > 200 else text,
        'text_length': len(text),
        'num_labels': len(predicted_labels),
        'predicted_classes': [],
        'keyword_matches': {}
    }
    
    for label_id in predicted_labels:
        if label_id in id2class:
            class_name = id2class[label_id]
            analysis['predicted_classes'].append({
                'id': label_id,
                'name': class_name
            })
            
            # Check keyword matches
            if class_name in class2keywords:
                matches = count_keyword_matches(text, class2keywords[class_name])
                if matches:
                    analysis['keyword_matches'][class_name] = matches
    
    return analysis

def find_success_cases(pid2text, tfidf_pred, gnn_pred, id2class, class2keywords, top_n=5):
    """Find examples with strong keyword matches (likely correct)"""
    success_cases = []
    
    for pid, labels in tfidf_pred.items():
        if pid not in pid2text:
            continue
        
        text = pid2text[pid]
        analysis = analyze_prediction(pid, text, labels, id2class, class2keywords)
        
        # Score: number of classes with keyword matches
        match_score = len(analysis['keyword_matches'])
        
        if match_score >= 2:  # At least 2 classes with keyword evidence
            analysis['confidence'] = 'high'
            analysis['match_score'] = match_score
            
            # Add GNN comparison
            if pid in gnn_pred:
                analysis['gnn_prediction'] = {
                    'labels': gnn_pred[pid],
                    'num_labels': len(gnn_pred[pid]),
                    'overlap': len(set(labels) & set(gnn_pred[pid])),
                    'classes': [{'id': lid, 'name': id2class.get(lid, 'Unknown')} 
                               for lid in gnn_pred[pid][:5]]
                }
            
            success_cases.append(analysis)
    
    # Sort by match score
    success_cases.sort(key=lambda x: x['match_score'], reverse=True)
    return success_cases[:top_n]

def find_failure_cases(pid2text, pid2labels, id2class, class2keywords, child2parents, top_n=5):
    """Find problematic examples (many labels, few keyword matches, hierarchy explosion)"""
    no_keyword_cases = []  # Type 1: No keyword matches
    hierarchy_explosion_cases = []  # Type 2: Too many labels
    
    for pid, labels in pid2labels.items():
        if pid not in pid2text:
            continue
        
        text = pid2text[pid]
        analysis = analyze_prediction(pid, text, labels, id2class, class2keywords)
        
        match_score = len(analysis['keyword_matches'])
        num_labels = len(labels)
        text_len = len(text)
        
        # Type 1: No keyword matches (ambiguous/generic text)
        if match_score == 0:
            problem_score = 3
            reasons = ["No keyword matches"]
            
            if text_len < 50:
                problem_score += 2
                reasons.append(f"Very short text ({text_len} chars)")
            
            analysis['confidence'] = 'low'
            analysis['problem_score'] = problem_score
            analysis['problems'] = reasons
            analysis['failure_type'] = 'no_keywords'
            no_keyword_cases.append(analysis)
        
        # Type 2: Hierarchy explosion (too many labels)
        elif num_labels >= 6:  # Changed from 5 to 6 for clearer cases
            problem_score = 2
            reasons = [f"Label explosion ({num_labels} labels)"]
            
            if match_score < 2:
                problem_score += 1
                reasons.append("Weak keyword evidence")
            
            # Check how many are from hierarchy vs original
            # Estimate: if match_score is low but num_labels high, likely hierarchy
            estimated_hierarchy_labels = num_labels - match_score
            if estimated_hierarchy_labels >= 3:
                reasons.append(f"~{estimated_hierarchy_labels} labels from hierarchy propagation")
            
            analysis['confidence'] = 'low'
            analysis['problem_score'] = problem_score
            analysis['problems'] = reasons
            analysis['failure_type'] = 'hierarchy_explosion'
            hierarchy_explosion_cases.append(analysis)
    
    # Sort both types
    no_keyword_cases.sort(key=lambda x: x['problem_score'], reverse=True)
    hierarchy_explosion_cases.sort(key=lambda x: x['num_labels'], reverse=True)
    
    # Return mix: first half from no_keywords, second half from hierarchy
    half = top_n // 2
    return no_keyword_cases[:half] + hierarchy_explosion_cases[:top_n-half]

# ==================== Main ====================

def main():
    print("="*60)
    print("CASE STUDY EXTRACTION")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    pid2text = load_corpus(TEST_CORPUS_PATH)
    id2class = load_classes(CLASSES_PATH)
    class2keywords = load_keywords(KEYWORDS_PATH)
    child2parents = load_hierarchy(HIERARCHY_PATH)
    
    print(f"  ✓ Test corpus: {len(pid2text):,} samples")
    print(f"  ✓ Classes: {len(id2class)}")
    
    # Load predictions
    print("\n[2/5] Loading predictions...")
    tfidf_pred = load_submission(TFIDF_SUBMISSION)
    gnn_pred = load_submission(GNN_SUBMISSION)
    
    print(f"  ✓ TF-IDF predictions: {len(tfidf_pred):,}")
    print(f"  ✓ GNN predictions: {len(gnn_pred):,}")
    
    # Find success cases (use TF-IDF as it has highest score, compare with GNN)
    print("\n[3/5] Finding success cases...")
    success_cases = find_success_cases(pid2text, tfidf_pred, gnn_pred, id2class, class2keywords, top_n=5)
    print(f"  ✓ Found {len(success_cases)} success examples")
    
    # Find failure cases
    print("\n[4/5] Finding failure cases...")
    failure_cases = find_failure_cases(pid2text, tfidf_pred, id2class, class2keywords, child2parents, top_n=5)
    print(f"  ✓ Found {len(failure_cases)} failure examples")
    
    # Save results
    print("\n[5/5] Saving results...")
    case_study = {
        'success_cases': success_cases,
        'failure_cases': failure_cases,
        'summary': {
            'total_test_samples': len(pid2text),
            'success_examples': len(success_cases),
            'failure_examples': len(failure_cases)
        }
    }
    
    with open(CASE_STUDY_PATH, 'w', encoding='utf-8') as f:
        json.dump(case_study, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Case study saved to: {CASE_STUDY_PATH}")
    
    # Print examples
    print("\n" + "="*60)
    print("SUCCESS CASE #1 (Strong keyword evidence)")
    print("="*60)
    if success_cases:
        case = success_cases[0]
        print(f"Product ID: {case['pid']}")
        print(f"Text: {case['text']}")
        print(f"\nTF-IDF Predicted ({case['num_labels']} labels):")
        for cls in case['predicted_classes'][:3]:
            print(f"  - {cls['id']}: {cls['name']}")
        print(f"\nKeyword matches:")
        for cls_name, matches in list(case['keyword_matches'].items())[:3]:
            print(f"  - {cls_name}: {', '.join(matches)}")
        
        # GNN comparison
        if 'gnn_prediction' in case:
            gnn = case['gnn_prediction']
            print(f"\nGNN Predicted ({gnn['num_labels']} labels):")
            for cls in gnn['classes'][:3]:
                print(f"  - {cls['id']}: {cls['name']}")
            print(f"  Overlap: {gnn['overlap']}/{case['num_labels']} labels match")
    
    print("\n" + "="*60)
    print("FAILURE CASES")
    print("="*60)
    
    # Print both types if available
    no_kw = [c for c in failure_cases if c.get('failure_type') == 'no_keywords']
    hierarchy = [c for c in failure_cases if c.get('failure_type') == 'hierarchy_explosion']
    
    if no_kw:
        print("\nType 1: No Keyword Matches (Ambiguous Text)")
        print("-" * 60)
        case = no_kw[0]
        print(f"Product ID: {case['pid']}")
        print(f"Text: {case['text']}")
        print(f"\nPredicted classes ({case['num_labels']}):")
        for cls in case['predicted_classes'][:3]:
            print(f"  - {cls['id']}: {cls['name']}")
        print(f"\nProblems:")
        for problem in case.get('problems', []):
            print(f"  - {problem}")
    
    if hierarchy:
        print("\nType 2: Hierarchy Explosion (Too Many Labels)")
        print("-" * 60)
        case = hierarchy[0]
        print(f"Product ID: {case['pid']}")
        print(f"Text: {case['text']}")
        print(f"\nPredicted classes ({case['num_labels']}):")
        for cls in case['predicted_classes'][:6]:
            print(f"  - {cls['id']}: {cls['name']}")
        print(f"\nProblems:")
        for problem in case.get('problems', []):
            print(f"  - {problem}")
        print(f"\nKeyword matches: {len(case['keyword_matches'])}")
    
    print("\n" + "="*60)
    print("✅ DONE! Use these examples for your report")
    print("="*60)

if __name__ == "__main__":
    main()
