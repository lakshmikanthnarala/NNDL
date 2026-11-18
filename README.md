# SignSpeak – AI-Powered Sign Language to Text Translator

SignSpeak is an academic Streamlit application that classifies **single, static ASL alphabet handshapes** with a TensorFlow/Keras convolutional neural network (CNN). It uses the real Sign Language MNIST CSV dataset and computes reported accuracy and visualizations from training and held-out test evaluation. Predictions are not hardcoded.

## Problem statement and objectives

Communication across spoken and signed languages can be difficult. This project demonstrates a deep-learning pipeline that maps an image of an isolated ASL alphabet gesture to a letter, then lets the user explicitly compose letters into text.

- Train a 24-class CNN on Sign Language MNIST.
- Preserve the dataset's non-contiguous label mapping (label 9 / J is absent).
- Offer webcam snapshot and PNG/JPG/JPEG upload prediction.
- Display the predicted letter, model confidence, and preprocessed image.
- Let users add, space, delete, copy, download, and clear composed text.
- Show measured training, validation, and test performance.

## Technologies

Python, TensorFlow/Keras, CNN, Streamlit, OpenCV, NumPy, Pandas, Pillow, scikit-learn, Matplotlib, and Seaborn.

## Dataset setup

1. Download `sign_mnist_train.csv` and `sign_mnist_test.csv` from the [Sign Language MNIST Kaggle dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
2. Place both CSV files in the project's `dataset/` directory:

   ```text
   dataset/
   ├── sign_mnist_train.csv
   └── sign_mnist_test.csv
   ```

3. If Kaggle requires account authentication, use Kaggle's own download instructions or download the files in a browser. Do not commit credentials or dataset files to a public repository.

The CSVs contain a `label` column and 784 pixel columns (`pixel1`–`pixel784`). Labels cover integers 0–24 except 9: 0–8 map to A–I and 10–24 map to K–Y. J and Z are excluded because they require movement. The training code remaps the remaining labels to contiguous output indices 0–23.

## Install and run on Windows

Use Python 3.10–3.11 for the pinned TensorFlow compatibility range. In PowerShell, from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run commands using the virtual-environment interpreter directly, for example `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

Put both dataset CSVs in `dataset/`, then train the model:

```powershell
python train_model.py
```

Training evaluates the best validation checkpoint on the separate Kaggle test CSV. When training succeeds, it writes:

- `models/sign_language_model.keras` — actual trained model
- `models/class_mapping.json` — dataset-label to letter and output-index mapping
- `outputs/evaluation_results.json` — measured metrics and sample counts
- `outputs/training_history.json` — recorded training/validation metrics by epoch
- `outputs/accuracy.png`, `outputs/loss.png`, `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`, `outputs/model_summary.txt`

Start the website by double-clicking `run_website.bat`, or run this command from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the local application in your browser at `http://localhost:8501`. Keep the terminal window open while using the site. The website can open before training, but predictions require the trained model; add the dataset CSVs and run `python train_model.py` first. Allow camera access to use **Webcam capture**. The app processes one captured image only; it does not claim continuous real-time recognition. Image upload accepts PNG, JPG, and JPEG.

## Deploy as a website

The repository includes a Docker configuration and hosted-platform startup files. The trained model and class mapping are committed under `models/`, so no training step is required by the deployed website.

Build and run with Docker:

```powershell
docker build -t signspeak .
docker run --rm -p 8501:8501 -e PORT=8501 signspeak
```

Then open `http://localhost:8501`.

For Streamlit Community Cloud, create a new app from this repository and set the main file to `app.py`. For Render, use `render.yaml` or create a Docker web service; for Railway or similar platforms, the included `Dockerfile` and `PORT` handling work directly. Streaming camera input requires HTTPS, which hosted platforms normally provide.

The same install, training, and run commands work in Google Colab after uploading or mounting the two CSV files into `dataset/`. In Colab, use `!python train_model.py` and `!streamlit run app.py`; expose Streamlit with an approved Colab-compatible tunnel if a browser preview is needed.

## Test

After training, verify label mapping and reload a saved model:

```powershell
python -m unittest discover -s tests -v
```

The model smoke test is skipped if the model has not yet been trained. It checks model loading and prediction shape/confidence, not expected accuracy for an arbitrary image. To validate an ASL image manually, open **Live & Upload**, provide a clear isolated handshape, and click **Predict**. Webcam hardware/browser permissions and the uploaded image itself are needed for interactive checks.

## Project structure

```text
SignSpeak/
├── app.py
├── train_model.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── dataset/                 # Add the downloaded CSV files here
├── models/                  # Trained model and class mapping are generated
├── outputs/                 # Evaluation metrics and plots are generated
├── tests/
│   ├── test_mapping.py
│   └── test_model_smoke.py
└── utils/
    ├── __init__.py
    └── class_mapping.py
```

## Method and expected results

The CSV pixel rows are reshaped to 28 × 28 × 1 and normalized to [0, 1]. A stratified 15% split of the training CSV is used for validation. Small rotations, translations, and zooms augment training examples without horizontal flipping. The CNN contains 32- and 64-filter convolution/pooling blocks, a 128-neuron dense layer, dropout, and a 24-unit softmax output. Adam and sparse categorical cross-entropy are used, with a maximum of 20 epochs, batch size 64, best-validation checkpointing, and early stopping.

The performance dashboard only shows the values and charts generated by a completed training run; no accuracy is promised in advance. Results depend on the dataset, runtime, and training environment.

Completely blank/low-contrast images are rejected. The interface also withholds letters when the highest softmax score is below 25%; this is a conservative usability heuristic, not a statistically calibrated rejection threshold.

## Limitations and future work

This is a static alphabet classifier, not a complete sign-language translator. It cannot recognize J/Z motion, continuous signing, grammar, facial expression, or sentence meaning. Sign Language MNIST images are small, centered grayscale examples; real webcam backgrounds, lighting, hand orientation, and camera framing differ and may reduce accuracy. The preprocessing is a lightweight image crop/resize, not a production-grade hand detector. Future work could add robust hand localization, diverse real-world data, temporal models for moving signs, and user testing.
