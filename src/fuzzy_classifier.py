#!/usr/bin/env python3
"""
MEA Spike Detection - Explainable Fuzzy Classifier
Author: Adham Aboulkheir | University of Essex | PhD Research
Published: WCCI FUZZ-IEEE 2026
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class FuzzyMF:
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
    def compute(self, x):
        x = np.atleast_1d(np.array(x, dtype=float))
        left  = (x - self.a) / (self.b - self.a + 1e-9)
        right = (self.c - x) / (self.c - self.b + 1e-9)
        return np.clip(np.minimum(left, right), 0, 1)


class FuzzyRule:
    def __init__(self, antecedents, consequent, weight=1.0, confidence=1.0, support=0.0):
        self.antecedents = antecedents
        self.consequent  = consequent
        self.weight      = weight
        self.confidence  = confidence
        self.support     = support

    def matching_degree(self, x, mfs):
        degree = 1.0
        for eid, tidx in self.antecedents.items():
            degree = min(degree, mfs[eid][tidx].compute(x[eid]))
        return degree

    def to_natural_language(self, term_names=None):
        if term_names is None:
            term_names = ["LOW", "MEDIUM", "HIGH"]
        conds = [f"Electrode_{eid+1} is {term_names[tidx]}" for eid, tidx in self.antecedents.items()]
        label = "SPIKE" if self.consequent == 1 else "NO-SPIKE"
        return f"IF {' AND '.join(conds)} -> {label} (Conf: {self.confidence:.1%}, Support: {self.support:.1%})"


class MEASpikeClassifier(BaseEstimator, ClassifierMixin):
    """
    XAI Fuzzy Rule-Based Classifier for MEA Spike Detection.
    Paper: 'A Fuzzy-Based Approach for Interpretable Spike Detection in Living Neural Biocomputers'
    WCCI FUZZ-IEEE 2026
    """
    def __init__(self, n_terms=3, n_selected_electrodes=20, random_state=42):
        self.n_terms = n_terms
        self.n_selected_electrodes = n_selected_electrodes
        self.random_state = random_state
        self.rules_ = []
        self.classes_ = np.array([0, 1])

    def _build_mfs(self, n_features):
        mfs = []
        for _ in range(n_features):
            mfs.append([
                FuzzyMF(0.0, 0.0, 0.5),
                FuzzyMF(0.0, 0.5, 1.0),
                FuzzyMF(0.5, 1.0, 1.0),
            ])
        return mfs

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)
        importances = np.array([abs(np.corrcoef(X_s[:, i], y)[0, 1]) for i in range(X_s.shape[1])])
        importances = np.nan_to_num(importances)
        self.selected_electrodes_ = np.argsort(importances)[::-1][:self.n_selected_electrodes]
        X_sel = X_s[:, self.selected_electrodes_]
        self.mfs_ = self._build_mfs(X_sel.shape[1])
        self._generate_rules(X_sel, y)
        return self

    def _generate_rules(self, X, y):
        self.rules_ = []
        n = X.shape[1]
        for cl in [0, 1]:
            Xc = X[y == cl]
            for i in range(min(n, 8)):
                for j in range(i+1, min(n, 8)):
                    for ti in range(self.n_terms):
                        for tj in range(self.n_terms):
                            md_i = self.mfs_[i][ti].compute(Xc[:, i])
                            md_j = self.mfs_[j][tj].compute(Xc[:, j])
                            sup = np.minimum(md_i, md_j).mean()
                            if sup > 0.05:
                                all_i = self.mfs_[i][ti].compute(X[:, i])
                                all_j = self.mfs_[j][tj].compute(X[:, j])
                                all_m = np.minimum(all_i, all_j)
                                correct = np.sum(all_m[y == cl])
                                conf = correct / (np.sum(all_m) + 1e-9)
                                if conf > 0.7:
                                    self.rules_.append(FuzzyRule({i: ti, j: tj}, cl, conf*sup, conf, sup))
        self.rules_.sort(key=lambda r: r.weight, reverse=True)
        self.rules_ = self.rules_[:337]

    def predict_proba(self, X):
        X_s = self.scaler_.transform(X)
        X_sel = X_s[:, self.selected_electrodes_]
        proba = np.zeros((len(X), 2))
        for idx, sample in enumerate(X_sel):
            scores = {0: 0.0, 1: 0.0}
            for rule in self.rules_:
                md = rule.matching_degree(sample, self.mfs_)
                scores[rule.consequent] += md * rule.weight
            total = sum(scores.values()) + 1e-9
            proba[idx, 0] = scores[0] / total
            proba[idx, 1] = scores[1] / total
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_rules(self, n=10):
        return [r.to_natural_language() for r in self.rules_[:n]]


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report
    print("MEA Spike Detection - Fuzzy Classifier Demo")
    print("=" * 50)
    np.random.seed(42)
    X, y = make_classification(n_samples=5000, n_features=142, n_informative=20,
                                weights=[0.6, 0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    clf = MEASpikeClassifier(n_selected_electrodes=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Selected electrodes: {clf.selected_electrodes_[:5]}... (top 5 of 20)")
    print(f"Active rules: {len(clf.rules_)}")
    print(f"F1-Score: {f1:.4f}")
    print("\nTop 5 Rules:")
    for r in clf.get_rules(5):
        print(f"  {r}")
