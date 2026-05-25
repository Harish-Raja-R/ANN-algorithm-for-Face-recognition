# PCA with ANN Face Recognition Report

## Dataset
- Enrolled identities: Aamir, Ajay
- Imposter identities: Akshay
- Training samples: 60
- Test samples, including imposters: 70

## Method
Images were converted to grayscale, resized, flattened into column vectors, mean centered, and projected into an eigenface basis computed from the surrogate covariance matrix. The projected signatures were classified with a one-hidden-layer ANN trained by backpropagation. Samples with low ANN confidence or large PCA-space nearest-neighbor distance were rejected as UNKNOWN.

## Accuracy vs k

| k | overall accuracy | enrolled accuracy | imposter rejection |
|---:|---:|---:|---:|
| 5 | 0.4286 | 0.4250 | 0.4333 |
| 10 | 0.4571 | 0.5250 | 0.3667 |
| 15 | 0.6000 | 0.7750 | 0.3667 |
| 20 | 0.6000 | 0.8000 | 0.3333 |
| 25 | 0.5571 | 0.7750 | 0.2667 |
| 30 | 0.5714 | 0.6750 | 0.4333 |

## Best Run
- Best k: 15
- Overall accuracy: 0.6000
- Enrolled-person accuracy: 0.7750
- Imposter rejection accuracy: 0.3667

Generated artifacts: `accuracy_vs_k.png`, `accuracy_vs_k.csv`, `confusion_matrix.csv`, `predictions.csv`, and the saved `.npz` model.