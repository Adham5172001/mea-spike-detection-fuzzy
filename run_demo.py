"""
MEA Spike Detection — Full End-to-End Demo
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.fuzzy_classifier import MEASpikeClassifier
from src.evaluate import evaluate_across_chips, print_evaluation_report


def main():
    print("=" * 60)
    print("MEA SPIKE DETECTION — FUZZY CLASSIFIER DEMO")
    print("Paper: A Fuzzy-Based Approach for Interpretable Spike")
    print("       Detection in Living Neural Biocomputers")
    print("Conference: WCCI FUZZ-IEEE 2026, Maastricht")
    print("Author: Adham Aboulkheir | University of Essex")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate 6 biologically diverse MEA chips
    print("\n[1/4] Generating simulated MEA data (6 chips)...")
    chips_X, chips_y = [], []
    for chip_id in range(6):
        X, y = make_classification(
            n_samples=5000, n_features=142, n_informative=20,
            n_redundant=10, n_clusters_per_class=2,
            weights=[0.6, 0.4], random_state=chip_id * 7
        )
        chips_X.append(X)
        chips_y.append(y)
        print(f"  Chip {chip_id+1}: {X.shape[0]} samples, "
              f"{y.sum()} spikes ({y.mean():.1%})")
    
    # Train on chips 1-4, test on chips 5-6
    X_train = np.vstack(chips_X[:4])
    y_train = np.concatenate(chips_y[:4])
    
    print(f"\n[2/4] Training fuzzy classifier...")
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} electrodes")
    
    clf = MEASpikeClassifier(
        n_terms=3,
        n_selected_electrodes=20,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print(f"  Selected electrodes: {clf.selected_electrodes_[:5]}... (top 5 of 20)")
    print(f"  Active rules generated: {len(clf.rules_)}")
    
    # Evaluate on all 6 chips
    print("\n[3/4] Evaluating across all 6 chips...")
    all_y_true, all_y_pred = [], []
    for chip_id, (X_chip, y_chip) in enumerate(zip(chips_X, chips_y)):
        y_pred = clf.predict(X_chip)
        all_y_true.append(y_chip)
        all_y_pred.append(y_pred)
        f1 = f1_score(y_chip, y_pred, average="weighted")
        print(f"  Chip {chip_id+1}: F1={f1:.4f}")
    
    results = evaluate_across_chips(all_y_true, all_y_pred)
    print_evaluation_report(results)
    
    # Show interpretable rules
    print("\n[4/4] Interpretable Rules (Top 10):")
    print("-" * 60)
    for i, rule in enumerate(clf.get_rules(10), 1):
        print(f"  Rule {i:2d}: {rule}")
    
    # Single prediction explanation
    sample = chips_X[0][0]
    label, conf, explanations = clf.predict_explain(sample)
    print(f"\nSingle Prediction Example:")
    print(f"  Prediction: {label} (confidence: {conf:.1%})")
    print("  Rules that fired:")
    for exp in explanations[:3]:
        print(f"    {exp}")
    
    print("\n✓ Demo complete!")
    print(f"  Mean F1-Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"  Mean GM:       {results['mean_gm']:.4f} ± {results['std_gm']:.4f}")


if __name__ == "__main__":
    main()
