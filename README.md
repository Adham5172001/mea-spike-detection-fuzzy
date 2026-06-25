# MEA Spike Detection — Explainable Fuzzy Classifier

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Conference](https://img.shields.io/badge/WCCI%20FUZZ--IEEE-2026-red)](https://attend.ieee.org/wcci-2026/)
[![PhD](https://img.shields.io/badge/University%20of%20Essex-PhD%20Research-purple)](https://essex.ac.uk)

> **Associated Publication**: *"A Fuzzy-Based Approach for Interpretable Spike Detection in Living Neural Biocomputers"*  
> Accepted at **WCCI FUZZ-IEEE 2026**, Maastricht, Netherlands

An Explainable AI (XAI) fuzzy rule-based classification system for detecting neural spikes in Multi-Electrode Array (MEA) recordings from living neural biocomputers. Unlike black-box deep learning models, this system produces human-readable IF-THEN rules that reveal *which* electrodes and *what* firing patterns are associated with spike events.

## Background

Living neural biocomputers are biological systems where living neurons are cultured on MEA chips and used for computation. Detecting and classifying neural spikes (action potentials) in these recordings is critical for understanding biological computation, drug safety testing, and neuroprosthetics.

Standard deep learning approaches achieve high accuracy but provide no insight into *why* a spike was detected — a critical limitation for scientific understanding and clinical applications. This work addresses that gap with an interpretable fuzzy system.

## System Architecture

```
MEA Recording Data (142 electrodes × N time steps)
        │
  Feature Extraction
  (Firing rate vectors per electrode per time window)
        │
  ANNIGMA Feature Selection
  (Neural network importance scoring: 142 → 20 electrodes)
        │
  Fuzzy Membership Function Generation
  (LOW / MEDIUM / HIGH per selected electrode)
        │
  Genetic Algorithm Rule Optimisation
  (Evolves IF-THEN rule base over 30 generations)
        │
  Fuzzy Inference & Classification
  (SPIKE / NO-SPIKE with confidence score)
        │
  Interpretable Rule Output
  e.g. "IF Electrode_13 is MEDIUM AND Electrode_17 is MEDIUM → SPIKE (Conf: 100%)"
```

## Key Results

| Metric | Value |
|--------|-------|
| F1-Score | **97.74%** |
| Geometric Mean (GM) | **96.14%** |
| Validated across | **6 biologically diverse chips** |
| Active rules | **337** interpretable IF-THEN rules |
| Selected electrodes | **20 out of 142** |

## Example Rules Generated

```
Rule 1: IF Electrode_13 is MEDIUM AND Electrode_17 is MEDIUM → SPIKE
        Confidence: 100.0% | Support: 12.3%

Rule 2: IF Electrode_7 is HIGH AND Electrode_22 is LOW → NO-SPIKE
        Confidence: 98.7% | Support: 8.1%

Rule 3: IF Electrode_17 is HIGH → SPIKE
        Confidence: 95.2% | Support: 15.7%
```

*Note: Electrode 17 appears in 47% of spike-detection rules, revealing it as a critical spatial hub in the neural network.*

## Dataset

Data sourced from the **Laboratory of Biosensors and Bioelectronics (LBB), ETH Zurich** (Küchler et al., 2025). HD-MEA recordings from engineered biological neural networks on CMOS chips.

- **Reference**: Küchler et al. (2025), *Frontiers in Computational Neuroscience*
- **Data repository**: [ETH Zurich Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/4d2c4948-ae69-4be6-b66b-366e1ae1aaf0)

## Installation

```bash
git clone https://github.com/Adham5172001/mea-spike-detection-fuzzy.git
cd mea-spike-detection-fuzzy
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Run full pipeline
python main.py --data_dir data/ --n_electrodes 20 --ga_generations 30

# Inspect generated rules
python inspect_rules.py --model_path models/best_classifier.pkl
```

## Usage

```python
from fuzzy_classifier import MEASpikeClassifier

# Load trained classifier
clf = MEASpikeClassifier.load("models/best_classifier.pkl")

# Classify a new recording window
firing_rates = extract_features(mea_recording)  # shape: (142,)
prediction, confidence, rules_fired = clf.predict_explain(firing_rates)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.1%}")
print("Rules that fired:")
for rule in rules_fired:
    print(f"  {rule}")
```

## Project Structure

```
mea-spike-detection-fuzzy/
├── src/
│   ├── feature_extraction.py    # Firing rate computation
│   ├── annigma.py               # Neural network feature selection
│   ├── fuzzy_system.py          # Membership functions & inference
│   ├── genetic_algorithm.py     # GA rule optimisation
│   └── classifier.py            # Main classifier class
├── scripts/
│   ├── download_data.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{aboulkheir2026fuzzy,
  title={A Fuzzy-Based Approach for Interpretable Spike Detection in Living Neural Biocomputers},
  author={Aboulkheir, Adham and Barros, Michael and Hagras, Hani},
  booktitle={2026 IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026},
  organization={IEEE}
}
```

## License

MIT License
