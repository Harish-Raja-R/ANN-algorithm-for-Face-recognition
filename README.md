# PCA + ANN Face Recognition

This project implements the assignment described in `project_lyst7052-1.pdf`.
It builds an eigenfaces feature extractor with PCA and trains a small
back-propagation neural network to recognize enrolled people.

## Dataset

The expected dataset layout is:

```text
dataset/dataset/faces/
  Aamir/
  Ajay/
  Akshay/
```

The default experiment uses `Aamir` and `Ajay` as enrolled identities and
`Akshay` as an imposter identity, because imposters must not belong to the
training set.

## Run

Use the bundled Python runtime available in Codex:

```powershell
python install -r requirements.py
cd src
python pca_ann_face_recognition.py
```

Outputs are written to `outputs/`:

- `accuracy_vs_k.csv`
- `accuracy_vs_k.png`
- `confusion_matrix.csv`
- `predictions.csv`
- `model_k_*.npz`
- `project_report.md`

## Method Summary

1. Convert every face image into a grayscale column vector.
2. Compute the mean face.
3. Subtract the mean from each training image.
4. Compute the surrogate covariance matrix `Delta.T @ Delta`.
5. Sort eigenvectors by descending eigenvalue.
6. Generate eigenfaces by projecting the selected eigenvectors back into image
   space.
7. Project each image into the eigenface space to get a compact signature.
8. Train a one-hidden-layer ANN using backpropagation.
9. Reject low-confidence or distant samples as `UNKNOWN`.

