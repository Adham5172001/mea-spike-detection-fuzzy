"""
Unit Tests for MEA Spike Detection Fuzzy Classifier
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fuzzy_classifier import MEASpikeClassifier
from src.feature_extraction import bandpass_filter, detect_spikes, compute_firing_rates
from src.evaluate import geometric_mean, evaluate_classifier


def test_bandpass_filter():
    """Test that bandpass filter preserves shape and reduces noise."""
    data = np.random.normal(0, 10, (10, 20000))
    filtered = bandpass_filter(data, fs=20000)
    assert filtered.shape == data.shape
    # Filtered signal should have lower variance (noise removed)
    assert filtered.std() < data.std()
    print("  ✓ test_bandpass_filter")


def test_detect_spikes():
    """Test spike detection returns correct structure."""
    np.random.seed(42)
    data = np.random.normal(0, 5, (5, 10000))
    # Add clear spikes
    data[0, [1000, 3000, 5000, 7000]] = -80
    
    spike_times = detect_spikes(data, fs=10000, threshold_factor=5.0)
    assert len(spike_times) == 5
    assert len(spike_times[0]) >= 1  # Should detect spikes on electrode 0
    print("  ✓ test_detect_spikes")


def test_compute_firing_rates():
    """Test firing rate computation shape and values."""
    spike_times = [np.array([0.5, 1.5, 2.5]) for _ in range(10)]
    rates = compute_firing_rates(spike_times, duration=5.0, window_size=1.0)
    assert rates.shape == (10, 5)
    assert rates.min() >= 0
    print("  ✓ test_compute_firing_rates")


def test_classifier_fit_predict():
    """Test that classifier fits and predicts correctly."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=30, n_informative=10,
                                weights=[0.6, 0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    clf = MEASpikeClassifier(n_selected_electrodes=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})
    f1 = f1_score(y_test, y_pred, average="weighted")
    assert f1 > 0.5, f"F1 too low: {f1}"
    print(f"  ✓ test_classifier_fit_predict (F1={f1:.3f})")


def test_geometric_mean():
    """Test geometric mean computation."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect
    gm = geometric_mean(y_true, y_pred)
    assert abs(gm - 1.0) < 1e-6, f"Expected 1.0, got {gm}"
    
    y_pred_bad = np.array([0, 0, 0, 0, 0, 0])  # All negative
    gm_bad = geometric_mean(y_true, y_pred_bad)
    assert gm_bad == 0.0
    print("  ✓ test_geometric_mean")


def test_classifier_rules():
    """Test that classifier generates interpretable rules."""
    from sklearn.datasets import make_classification
    
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=8,
                                weights=[0.6, 0.4], random_state=42)
    clf = MEASpikeClassifier(n_selected_electrodes=8, random_state=42)
    clf.fit(X, y)
    
    rules = clf.get_rules(5)
    assert len(rules) <= 5
    assert all(isinstance(r, str) for r in rules)
    assert all("SPIKE" in r or "NO-SPIKE" in r for r in rules)
    print(f"  ✓ test_classifier_rules ({len(clf.rules_)} rules generated)")


if __name__ == "__main__":
    print("Running MEA Classifier Unit Tests")
    print("=" * 40)
    test_bandpass_filter()
    test_detect_spikes()
    test_compute_firing_rates()
    test_classifier_fit_predict()
    test_geometric_mean()
    test_classifier_rules()
    print("\n✓ All tests passed!")
