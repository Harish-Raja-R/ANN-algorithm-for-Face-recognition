from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
UNKNOWN_LABEL = "UNKNOWN"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    train_paths: list[str]
    x_test: np.ndarray
    y_test: np.ndarray
    test_paths: list[str]
    labels: list[str]
    enrolled_labels: list[str]
    imposter_labels: list[str]


@dataclass
class PcaModel:
    mean: np.ndarray
    eigenfaces: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    train_features: np.ndarray
    train_labels: np.ndarray


class SimpleANN:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        learning_rate: float = 0.03,
        l2: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / max(1, input_dim)), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / max(1, hidden_dim)), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return hidden, probs

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        n_samples = x.shape[0]
        y_one_hot = np.eye(int(y.max()) + 1)[y]

        for _ in range(epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y_one_hot[idx]

                hidden, probs = self._forward(xb)
                d_logits = (probs - yb) / xb.shape[0]

                grad_w2 = hidden.T @ d_logits + self.l2 * self.w2
                grad_b2 = d_logits.sum(axis=0)
                d_hidden = (d_logits @ self.w2.T) * (1.0 - hidden * hidden)
                grad_w1 = xb.T @ d_hidden + self.l2 * self.w1
                grad_b1 = d_hidden.sum(axis=0)

                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * grad_b2
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[1]


def read_image(path: Path, image_size: int) -> np.ndarray | None:
    try:
        if path.stat().st_size == 0:
            return None
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float64).reshape(-1) / 255.0
    except Exception:
        return None


def load_faces(data_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    class_names: list[str] = []

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        loaded_for_class = 0
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            row = read_image(image_path, image_size)
            if row is None:
                continue
            rows.append(row)
            labels.append(class_dir.name)
            paths.append(str(image_path))
            loaded_for_class += 1
        if loaded_for_class:
            class_names.append(class_dir.name)

    if not rows:
        raise ValueError(f"No readable face images found in {data_dir}")

    return np.vstack(rows), np.asarray(labels), paths, class_names


def make_split(
    x: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    class_names: list[str],
    train_ratio: float,
    imposter_labels: list[str],
    seed: int,
) -> DatasetSplit:
    rng = random.Random(seed)
    enrolled_labels = [label for label in class_names if label not in imposter_labels]
    if len(enrolled_labels) < 2:
        raise ValueError("At least two enrolled classes are required for ANN training.")

    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in enrolled_labels:
        idx = [i for i, value in enumerate(y) if value == label]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * train_ratio)))
        cut = min(cut, len(idx) - 1)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])

    for label in imposter_labels:
        test_idx.extend(i for i, value in enumerate(y) if value == label)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return DatasetSplit(
        x_train=x[train_idx],
        y_train=y[train_idx],
        train_paths=[paths[i] for i in train_idx],
        x_test=x[test_idx],
        y_test=np.asarray([UNKNOWN_LABEL if y[i] in imposter_labels else y[i] for i in test_idx]),
        test_paths=[paths[i] for i in test_idx],
        labels=enrolled_labels + [UNKNOWN_LABEL],
        enrolled_labels=enrolled_labels,
        imposter_labels=imposter_labels,
    )


def fit_pca(x_train: np.ndarray, y_train: np.ndarray, k: int) -> PcaModel:
    mean = x_train.mean(axis=0)
    centered = x_train - mean
    covariance = centered @ centered.T / max(1, x_train.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    selected = eigenvectors[:, order[:k]]
    eigenfaces = centered.T @ selected
    eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True) + 1e-12

    train_features = centered @ eigenfaces
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-8
    train_features = (train_features - feature_mean) / feature_std
    return PcaModel(mean, eigenfaces, feature_mean, feature_std, train_features, y_train)


def transform_pca(model: PcaModel, x: np.ndarray) -> np.ndarray:
    features = (x - model.mean) @ model.eigenfaces
    return (features - model.feature_mean) / model.feature_std


def nearest_enrolled_distance(train_features: np.ndarray, train_labels: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sqrt(((x[:, None, :] - train_features[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = distances.argmin(axis=1)
    return distances[np.arange(x.shape[0]), nearest_idx], train_labels[nearest_idx]


def leave_one_out_distance(features: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def evaluate_k(
    split: DatasetSplit,
    k: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + k)
    label_to_id = {label: i for i, label in enumerate(split.enrolled_labels)}
    y_train_id = np.asarray([label_to_id[label] for label in split.y_train], dtype=int)

    pca = fit_pca(split.x_train, split.y_train, k)
    ann = SimpleANN(k, hidden_dim, len(split.enrolled_labels), rng)
    ann.fit(pca.train_features, y_train_id, epochs=epochs, batch_size=batch_size, rng=rng)

    train_probs = ann.predict_proba(pca.train_features)
    train_conf = train_probs.max(axis=1)
    confidence_threshold = max(0.45, min(0.85, float(np.percentile(train_conf, 5) - 0.05)))

    train_dist = leave_one_out_distance(pca.train_features)
    distance_threshold = float(np.percentile(train_dist, 95) * 1.25)

    x_test_features = transform_pca(pca, split.x_test)
    probs = ann.predict_proba(x_test_features)
    pred_ids = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    pred_labels = np.asarray([split.enrolled_labels[i] for i in pred_ids], dtype=object)

    nearest_dist, _ = nearest_enrolled_distance(pca.train_features, split.y_train, x_test_features)
    reject = (pred_conf < confidence_threshold) | (nearest_dist > distance_threshold)
    pred_labels[reject] = UNKNOWN_LABEL

    accuracy = float((pred_labels == split.y_test).mean())
    known_mask = split.y_test != UNKNOWN_LABEL
    imposter_mask = split.y_test == UNKNOWN_LABEL
    known_accuracy = float((pred_labels[known_mask] == split.y_test[known_mask]).mean()) if known_mask.any() else 0.0
    imposter_rejection = float((pred_labels[imposter_mask] == UNKNOWN_LABEL).mean()) if imposter_mask.any() else 0.0

    return {
        "k": k,
        "accuracy": accuracy,
        "known_accuracy": known_accuracy,
        "imposter_rejection": imposter_rejection,
        "confidence_threshold": confidence_threshold,
        "distance_threshold": distance_threshold,
        "predictions": pred_labels,
        "probabilities": probs,
        "confidences": pred_conf,
        "distances": nearest_dist,
        "pca": pca,
        "ann": ann,
    }


def write_accuracy_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["k", "accuracy", "known_accuracy", "imposter_rejection", "confidence_threshold", "distance_threshold"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in writer.fieldnames})


def write_predictions_csv(path: Path, split: DatasetSplit, result: dict[str, object]) -> None:
    predictions = result["predictions"]
    confidences = result["confidences"]
    distances = result["distances"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "actual", "predicted", "ann_confidence", "nearest_distance"])
        for row in zip(split.test_paths, split.y_test, predictions, confidences, distances):
            writer.writerow(row)


def write_confusion_matrix(path: Path, labels: list[str], actual: np.ndarray, predicted: np.ndarray) -> None:
    label_to_id = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(actual, predicted):
        matrix[label_to_id[str(true_label)], label_to_id[str(pred_label)]] += 1

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 900, 560
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 60, 85
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)

    ks = [int(row["k"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    min_k, max_k = min(ks), max(ks)

    def point(k_value: int, acc: float) -> tuple[int, int]:
        x_span = max(1, max_k - min_k)
        x = plot_left + int((k_value - min_k) / x_span * (plot_right - plot_left))
        y = plot_bottom - int(acc * (plot_bottom - plot_top))
        return x, y

    for tick in range(0, 101, 20):
        y = plot_bottom - int((tick / 100) * (plot_bottom - plot_top))
        draw.line([plot_left - 5, y, plot_right, y], fill=(220, 220, 220))
        draw.text((25, y - 8), f"{tick}%", fill="black")

    points = [point(k, acc) for k, acc in zip(ks, accuracies)]
    if len(points) > 1:
        draw.line(points, fill=(30, 96, 175), width=4)
    for (x, y), k, acc in zip(points, ks, accuracies):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(30, 96, 175))
        draw.text((x - 16, y - 25), f"{acc:.2f}", fill="black")
        draw.text((x - 8, plot_bottom + 15), str(k), fill="black")

    draw.text((width // 2 - 120, 20), "Accuracy vs PCA k value", fill="black")
    draw.text((width // 2 - 35, height - 35), "k value", fill="black")
    draw.text((10, 20), "Accuracy", fill="black")
    image.save(path)


def save_model(path: Path, result: dict[str, object], labels: list[str]) -> None:
    pca: PcaModel = result["pca"]  # type: ignore[assignment]
    ann: SimpleANN = result["ann"]  # type: ignore[assignment]
    np.savez(
        path,
        labels=np.asarray(labels),
        mean=pca.mean,
        eigenfaces=pca.eigenfaces,
        feature_mean=pca.feature_mean,
        feature_std=pca.feature_std,
        train_features=pca.train_features,
        train_labels=pca.train_labels,
        w1=ann.w1,
        b1=ann.b1,
        w2=ann.w2,
        b2=ann.b2,
        confidence_threshold=result["confidence_threshold"],
        distance_threshold=result["distance_threshold"],
    )


def write_report(path: Path, split: DatasetSplit, rows: list[dict[str, object]], best: dict[str, object]) -> None:
    lines = [
        "# PCA with ANN Face Recognition Report",
        "",
        "## Dataset",
        f"- Enrolled identities: {', '.join(split.enrolled_labels)}",
        f"- Imposter identities: {', '.join(split.imposter_labels)}",
        f"- Training samples: {len(split.y_train)}",
        f"- Test samples, including imposters: {len(split.y_test)}",
        "",
        "## Method",
        "Images were converted to grayscale, resized, flattened into column vectors, mean centered, and projected into an eigenface basis computed from the surrogate covariance matrix. The projected signatures were classified with a one-hidden-layer ANN trained by backpropagation. Samples with low ANN confidence or large PCA-space nearest-neighbor distance were rejected as UNKNOWN.",
        "",
        "## Accuracy vs k",
        "",
        "| k | overall accuracy | enrolled accuracy | imposter rejection |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {float(row['accuracy']):.4f} | {float(row['known_accuracy']):.4f} | {float(row['imposter_rejection']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Run",
            f"- Best k: {best['k']}",
            f"- Overall accuracy: {float(best['accuracy']):.4f}",
            f"- Enrolled-person accuracy: {float(best['known_accuracy']):.4f}",
            f"- Imposter rejection accuracy: {float(best['imposter_rejection']):.4f}",
            "",
            "Generated artifacts: `accuracy_vs_k.png`, `accuracy_vs_k.csv`, `confusion_matrix.csv`, `predictions.csv`, and the saved `.npz` model.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA eigenfaces with a backpropagation ANN.")
    parser.add_argument("--data-dir", default="dataset/dataset/faces", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--image-size", default=64, type=int)
    parser.add_argument("--train-ratio", default=0.60, type=float)
    parser.add_argument("--imposter-labels", nargs="*", default=None)
    parser.add_argument("--k-values", nargs="*", type=int, default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--hidden-dim", default=24, type=int)
    parser.add_argument("--epochs", default=900, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = PROJECT_ROOT / args.data_dir
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, paths, class_names = load_faces(args.data_dir, args.image_size)
    imposter_labels = args.imposter_labels if args.imposter_labels else [class_names[-1]]
    split = make_split(x, y, paths, class_names, args.train_ratio, imposter_labels, args.seed)

    max_k = min(split.x_train.shape[0] - 1, split.x_train.shape[1])
    k_values = [k for k in args.k_values if 1 <= k <= max_k]
    if not k_values:
        raise ValueError(f"No valid k values. Choose values between 1 and {max_k}.")

    results = [
        evaluate_k(split, k, args.hidden_dim, args.epochs, args.batch_size, args.seed)
        for k in k_values
    ]
    best = max(results, key=lambda row: (float(row["accuracy"]), float(row["imposter_rejection"])))

    write_accuracy_csv(args.output_dir / "accuracy_vs_k.csv", results)
    write_plot(args.output_dir / "accuracy_vs_k.png", results)
    write_predictions_csv(args.output_dir / "predictions.csv", split, best)
    write_confusion_matrix(args.output_dir / "confusion_matrix.csv", split.labels, split.y_test, best["predictions"])
    save_model(args.output_dir / f"model_k_{best['k']}.npz", best, split.enrolled_labels)
    write_report(args.output_dir / "project_report.md", split, results, best)

    print("PCA + ANN face recognition complete")
    print(f"Loaded classes: {', '.join(class_names)}")
    print(f"Enrolled: {', '.join(split.enrolled_labels)}")
    print(f"Imposters: {', '.join(split.imposter_labels)}")
    print(f"Training samples: {len(split.y_train)}")
    print(f"Test samples: {len(split.y_test)}")
    print(f"Best k: {best['k']}")
    print(f"Best overall accuracy: {float(best['accuracy']):.4f}")
    print(f"Best enrolled accuracy: {float(best['known_accuracy']):.4f}")
    print(f"Best imposter rejection: {float(best['imposter_rejection']):.4f}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
