"""
ANNIGMA — Artificial Neural Network Input Gain Measurement Analysis
Feature selection for MEA electrode importance scoring.
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


class ANNIGMASelector:
    """
    ANNIGMA-based feature selection using neural network importance scoring.
    
    Trains a shallow MLP and uses permutation importance to rank electrodes
    by their contribution to spike classification.
    """
    
    def __init__(self, n_selected: int = 20, hidden_size: int = 50,
                 n_repeats: int = 10, random_state: int = 42):
        self.n_selected = n_selected
        self.hidden_size = hidden_size
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importance_scores_ = None
        self.selected_indices_ = None
        self.model_ = None
        self.scaler_ = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNIGMASelector":
        """
        Fit the ANNIGMA selector on training data.
        
        Parameters
        ----------
        X : (n_samples, n_electrodes) firing rate matrix
        y : (n_samples,) class labels
        
        Returns
        -------
        self
        """
        X_scaled = self.scaler_.fit_transform(X)
        
        # Train shallow MLP for importance scoring
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(self.hidden_size,),
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.model_.fit(X_scaled, y)
        
        # Compute permutation importance
        result = permutation_importance(
            self.model_, X_scaled, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring="f1_weighted"
        )
        
        self.importance_scores_ = result.importances_mean
        self.selected_indices_ = np.argsort(self.importance_scores_)[::-1][:self.n_selected]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select the top-k electrodes from X."""
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    def get_top_electrodes(self, n: int = None) -> np.ndarray:
        """Return indices of top-n most important electrodes."""
        n = n or self.n_selected
        return self.selected_indices_[:n]
    
    def importance_report(self) -> str:
        """Generate a human-readable importance report."""
        lines = ["ANNIGMA Electrode Importance Report", "=" * 40]
        for rank, idx in enumerate(self.selected_indices_[:10], 1):
            score = self.importance_scores_[idx]
            lines.append(f"  Rank {rank:2d}: Electrode_{idx+1:3d}  score={score:.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("ANNIGMA Feature Selection Demo")
    print("=" * 40)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=2000, n_features=142, n_informative=20,
                                n_redundant=10, weights=[0.6, 0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    selector = ANNIGMASelector(n_selected=20, hidden_size=50)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_sel.shape[1]}")
    print(f"\n{selector.importance_report()}")
    
    # Evaluate with selected features
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score
    
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_sel, y_train)
    f1 = f1_score(y_test, clf.predict(X_test_sel), average="weighted")
    print(f"\nF1-Score with {selector.n_selected} selected electrodes: {f1:.4f}")
