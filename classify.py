"""
classify.py
Simple Image Classification & Image Processing with OpenCV.

Pipeline:
    1. Load a labeled image dataset (digits 0-9).
    2. Preprocess with OpenCV: grayscale, resize, normalize.
    3. Apply image enhancement (brightness/contrast + filtering) and save a
       side-by-side comparison figure.
    4. Split into train/test sets (80/20).
    5. Train two classifiers: SVM and k-NN.
    6. Evaluate with accuracy + confusion matrix.
    7. Visualize sample predictions (predicted vs actual labels).

Dataset note:
    The task suggests CIFAR-10, but downloading it requires network access
    to external hosts that are not reachable in this environment. Instead,
    this script uses scikit-learn's bundled "digits" dataset (1,797 labeled
    8x8 grayscale images of handwritten digits 0-9). It needs no download,
    keeps the pipeline fully reproducible, and demonstrates every required
    step (OpenCV preprocessing, enhancement, SVM/k-NN training, evaluation).
    To use a different image folder dataset instead, see the
    `load_custom_dataset()` function below.
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "output"
IMG_SIZE = 32  # size (in pixels) each image is resized to for the model


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------
def load_digits_dataset():
    """Load the built-in sklearn digits dataset as a list of uint8 images."""
    digits = load_digits()
    # digits.images: (1797, 8, 8) float arrays with values 0-16
    images = digits.images
    labels = digits.target

    # Convert to standard 0-255 uint8 grayscale images (what a real
    # "collected" image dataset would look like straight off disk).
    images_uint8 = [
        cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for img in images
    ]
    class_names = [str(i) for i in range(10)]
    return images_uint8, labels, class_names


def load_custom_dataset(root_dir):
    """
    Optional: load a custom labeled dataset from a folder structure like:
        root_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
                ...
    Returns (images, labels, class_names). Not used by default, but provided
    so this script can be pointed at a real dataset (e.g. CIFAR-10 extracted
    to folders, or your own images) with minimal changes.
    """
    images, labels = [], []
    class_names = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            images.append(img)
            labels.append(label_idx)
    return images, np.array(labels), class_names


# ---------------------------------------------------------------------------
# 2. OpenCV preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(img, size=IMG_SIZE):
    """Grayscale -> resize -> normalize to [0, 1]. Returns a float32 array."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def enhance_image(img):
    """
    Apply brightness/contrast adjustment and a sharpening filter.
    Works on a uint8 grayscale image and returns a uint8 grayscale image.
    """
    # Brightness/contrast: new_pixel = alpha * pixel + beta
    alpha = 1.3  # contrast control (>1 increases contrast)
    beta = 20  # brightness control (>0 brightens)
    bright_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Sharpening filter (unsharp-mask style kernel)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(bright_contrast, -1, kernel)

    return sharpened


def save_enhancement_comparison(images_uint8, out_path, n=5):
    """Save a side-by-side figure of original vs enhanced images."""
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for i in range(n):
        original = images_uint8[i]
        enhanced = enhance_image(original)

        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(enhanced, cmap="gray")
        axes[1, i].set_title("Enhanced")
        axes[1, i].axis("off")

    plt.suptitle("Original vs. Enhanced (brightness/contrast + sharpening)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved enhancement comparison to: {out_path}")


# ---------------------------------------------------------------------------
# 3. Model training & evaluation
# ---------------------------------------------------------------------------
def build_feature_matrix(images_uint8):
    """Preprocess every image and flatten into a feature matrix."""
    features = [preprocess_image(img).flatten() for img in images_uint8]
    return np.array(features, dtype=np.float32)


def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    results = {}

    # Feature scaling helps both SVM and k-NN.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM": SVC(kernel="rbf", gamma="scale", C=10),
        "kNN": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name} accuracy: {acc:.4f}")

        # Save confusion matrix figure.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name} Confusion Matrix (accuracy={acc:.2%})")
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.lower()}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix to: {cm_path}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "y_pred": y_pred,
        }

    return results, scaler


def save_sample_predictions(X_test_raw_images, y_test, y_pred, class_names, out_path, n=10):
    """Show a few test images with predicted vs actual labels."""
    n = min(n, len(X_test_raw_images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = X_test_raw_images[i]
        pred_label = class_names[y_pred[i]]
        true_label = class_names[y_test[i]]
        correct = pred_label == true_label

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"Pred: {pred_label} | True: {true_label}",
            color="green" if correct else "red",
            fontsize=9,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Test Predictions (green = correct, red = wrong)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved sample predictions to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    images_uint8, labels, class_names = load_digits_dataset()
    print(f"Loaded {len(images_uint8)} images across {len(class_names)} classes.")

    # --- Image enhancement demo ---
    save_enhancement_comparison(
        images_uint8, os.path.join(OUTPUT_DIR, "enhancement_comparison.png")
    )

    # --- Preprocess all images into a feature matrix ---
    print("\nPreprocessing images (grayscale -> resize -> normalize)...")
    X = build_feature_matrix(images_uint8)
    y = labels

    # --- Train/test split (80/20), keep raw images aligned for visualization ---
    indices = np.arange(len(images_uint8))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} images | Test set: {len(X_test)} images")

    # --- Train & evaluate models ---
    print("\nTraining models...")
    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, class_names)

    # --- Save sample predictions using the better-performing model ---
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_result = results[best_name]
    print(f"\nBest model: {best_name} (accuracy={best_result['accuracy']:.4f})")

    test_raw_images = [images_uint8[i] for i in idx_test]
    save_sample_predictions(
        test_raw_images,
        y_test,
        best_result["y_pred"],
        class_names,
        os.path.join(OUTPUT_DIR, "sample_predictions.png"),
    )

    # --- Write a results summary file ---
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Image Classification Results\n")
        f.write("=============================\n\n")
        f.write(f"Dataset: sklearn digits (1797 images, classes 0-9)\n")
        f.write(f"Train/Test split: {len(X_train)} / {len(X_test)} (80/20, stratified)\n\n")
        for name, res in results.items():
            f.write(f"{name} accuracy: {res['accuracy']:.4f}\n")
        f.write(f"\nBest model: {best_name}\n")
    print(f"\nSaved results summary to: {summary_path}")


if __name__ == "__main__":
    main()
