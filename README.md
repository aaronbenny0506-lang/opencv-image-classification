# Simple Image Classification & Image Processing with OpenCV

A small end-to-end pipeline that uses **OpenCV** for image preprocessing/
enhancement and **SVM** / **k-NN** (scikit-learn) for classifying labeled
images.

## Dataset

The task suggested **CIFAR-10**, but downloading it requires reaching
external hosts (`cs.toronto.edu`) that aren't accessible from the environment
this was built in. Instead, this project uses scikit-learn's bundled
**digits dataset**: 1,797 labeled 8x8 grayscale images of handwritten digits
(0–9). It's a real labeled image classification dataset, needs no download
and exercises every part of the pipeline the task asks for (grayscale,
resize, normalize, enhancement, train/test split, SVM/k-NN, evaluation).

If you'd rather use CIFAR-10 or your own image folders, `classify.py`
includes a `load_custom_dataset(root_dir)` function that loads any dataset
laid out as `root_dir/<class_name>/<image files>`, swap it in for
`load_digits_dataset()` in `main()` and everything downstream (OpenCV
preprocessing, model training, evaluation) works unchanged.

## Pipeline / Steps Followed

1. **Load dataset** : `load_digits_dataset()` loads the 1,797 digit images
   and converts them to standard 0–255 `uint8` grayscale images.
2. **OpenCV preprocessing** (`preprocess_image`) :
   - Convert to grayscale (`cv2.cvtColor`, skipped if already grayscale)
   - Resize to a fixed 32×32 size (`cv2.resize`)
   - Normalize pixel values to `[0, 1]`
3. **Image enhancement** (`enhance_image`) :
   - Brightness/contrast adjustment via `cv2.convertScaleAbs`
     (`alpha=1.3` contrast, `beta=20` brightness)
   - Sharpening filter via `cv2.filter2D` with an unsharp-mask kernel
   - Saved as a side-by-side comparison: `output/enhancement_comparison.png`
4. **Train/test split** : 80% train / 20% test, stratified by class
   (`train_test_split`, `test_size=0.2`).
5. **Model training** : two classifiers trained on flattened, scaled pixel
   features:
   - **SVM** (`sklearn.svm.SVC`, RBF kernel)
   - **k-NN** (`sklearn.neighbors.KNeighborsClassifier`, k=5)
6. **Evaluation** :
   - Accuracy score for each model
   - Confusion matrix plotted for each model
     (`output/confusion_matrix_svm.png`, `output/confusion_matrix_knn.png`)
   - Sample test predictions visualized with predicted vs. actual labels
     (`output/sample_predictions.png`, green = correct, red = wrong)

## Results

| Model | Accuracy |
|-------|----------|
| SVM   | 98.06%   |
| k-NN  | 96.39%   |

SVM performed best on this dataset. Full numeric summary in
`output/results_summary.txt`.

## Project Structure

```
image-classification-opencv/
├── classify.py                 # main script (run this)
├── requirements.txt
├── README.md
└── output/
    ├── enhancement_comparison.png   # original vs enhanced images
    ├── confusion_matrix_svm.png
    ├── confusion_matrix_knn.png
    ├── sample_predictions.png       # predicted vs actual labels
    └── results_summary.txt          # accuracy numbers
```

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python classify.py
```

All output images and the results summary are written to the `output/`
folder.
