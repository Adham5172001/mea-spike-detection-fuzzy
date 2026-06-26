"""
Evaluation Metrics for MEA Spike Detection
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report, roc_auc_score)
from typing import Dict, List


def geometric_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Geometric Mean (GM) — balanced metric for imbalanced datasets.
    GM = sqrt(sensitivity × specificity)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    return float(np.sqrt(sensitivity * specificity))


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray = None,
                         chip_id: str = "all") -> Dict[str, float]:
    """
    Comprehensive evaluation of the fuzzy classifier.
    
    Returns
    -------
    metrics : dict with F1, GM, precision, recall, AUC-ROC
    """
    metrics = {
        "chip_id":    chip_id,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_spike":    f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_nospike":  f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "gm":          geometric_mean(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            metrics["auc_roc"] = 0.0
    return metrics


def evaluate_across_chips(y_true_list: List[np.ndarray],
                           y_pred_list: List[np.ndarray],
                           chip_ids: List[str] = None) -> Dict:
    """
    Evaluate classifier performance across multiple MEA chips.
    
    Returns per-chip metrics and aggregate statistics.
    """
    if chip_ids is None:
        chip_ids = [f"Chip_{i+1}" for i in range(len(y_true_list))]
    
    per_chip = []
    for y_true, y_pred, chip_id in zip(y_true_list, y_pred_list, chip_ids):
        metrics = evaluate_classifier(y_true, y_pred, chip_id=chip_id)
        per_chip.append(metrics)
    
    # Aggregate
    f1_scores = [m["f1_weighted"] for m in per_chip]
    gm_scores  = [m["gm"] for m in per_chip]
    
    aggregate = {
        "mean_f1":  np.mean(f1_scores),
        "std_f1":   np.std(f1_scores),
        "min_f1":   np.min(f1_scores),
        "max_f1":   np.max(f1_scores),
        "mean_gm":  np.mean(gm_scores),
        "std_gm":   np.std(gm_scores),
        "per_chip": per_chip,
    }
    return aggregate


def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("MEA SPIKE DETECTION — EVALUATION REPORT")
    print("=" * 60)
    
    if "per_chip" in results:
        print("\nPer-Chip Results:")
        print(f"  {'Chip':<12} {'F1':>8} {'GM':>8} {'Precision':>10} {'Recall':>8}")
        print("  " + "-" * 50)
        for m in results["per_chip"]:
            print(f"  {m['chip_id']:<12} {m['f1_weighted']:>8.4f} {m['gm']:>8.4f} "
                  f"{m['precision']:>10.4f} {m['recall']:>8.4f}")
        
        print(f"\n  Mean F1:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        print(f"  Mean GM:  {results['mean_gm']:.4f} ± {results['std_gm']:.4f}")
        print(f"  Range:    [{results['min_f1']:.4f}, {results['max_f1']:.4f}]")
    else:
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    
    print("Evaluation Module Demo")
    np.random.seed(42)
    
    # Simulate 6 chips
    all_y_true, all_y_pred = [], []
    for chip in range(6):
        X, y = make_classification(n_samples=1000, n_features=20, weights=[0.6, 0.4],
                                    random_state=chip)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = GradientBoostingClassifier(n_estimators=50, random_state=chip)
        clf.fit(X_train, y_train)
        all_y_true.append(y_test)
        all_y_pred.append(clf.predict(X_test))
    
    results = evaluate_across_chips(all_y_true, all_y_pred)
    print_evaluation_report(results)
