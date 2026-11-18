"""Train, evaluate, and save the SignSpeak CNN on Sign Language MNIST."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.class_mapping import (
    CLASSES,
    DATASET_LABEL_TO_CLASS,
    DATASET_LABELS,
    encode_dataset_labels,
)


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
TRAIN_CSV = DATASET_DIR / "sign_mnist_train.csv"
TEST_CSV = DATASET_DIR / "sign_mnist_test.csv"


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download sign_mnist_train.csv and sign_mnist_test.csv from "
            "https://www.kaggle.com/datasets/datamunge/sign-language-mnist "
            "and place both files in the dataset/ folder."
        )
    frame = pd.read_csv(path)
    expected_columns = {"label", *(f"pixel{i}" for i in range(1, 785))}
    if set(frame.columns) != expected_columns or len(frame.columns) != 785:
        raise ValueError(
            f"{path.name} must contain a label column and pixel1 through pixel784."
        )
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset(DATASET_LABELS):
        raise ValueError(f"{path.name} contains labels outside the 24 supported classes.")
    if set(np.unique(labels)) != set(DATASET_LABELS):
        raise ValueError(f"{path.name} must include all 24 supported classes.")
    images = frame.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError(f"{path.name} contains invalid pixel values.")
    images = images.reshape((-1, 28, 28, 1)) / 255.0
    return images, np.asarray(encode_dataset_labels(labels), dtype=np.int64)


def build_model():
    from tensorflow import keras

    layers = keras.layers
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.04, fill_mode="nearest"),
            layers.RandomTranslation(0.06, 0.06, fill_mode="nearest"),
            layers.RandomZoom(0.06, fill_mode="nearest"),
        ],
        name="training_augmentation",
    )
    inputs = keras.Input(shape=(28, 28, 1), name="grayscale_handshape")
    x = augmentation(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="letter_probabilities")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="signspeak_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plots(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SignSpeak training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SignSpeak training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss.png", dpi=160)
    plt.close()


def main() -> None:
    try:
        from tensorflow import keras
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is required. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_images, train_labels = load_dataset(TRAIN_CSV)
    test_images, test_labels = load_dataset(TEST_CSV)
    fit_images, validation_images, fit_labels, validation_labels = train_test_split(
        train_images,
        train_labels,
        test_size=0.15,
        random_state=42,
        stratify=train_labels,
    )

    model = build_model()
    checkpoint_path = MODEL_DIR / "sign_language_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        fit_images,
        fit_labels,
        validation_data=(validation_images, validation_labels),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    model = keras.models.load_model(checkpoint_path)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    probabilities = model.predict(test_images, batch_size=64, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    actual_letters = [CLASSES[index] for index in test_labels]
    predicted_letters = [CLASSES[index] for index in predicted_labels]
    report = classification_report(
        actual_letters,
        predicted_letters,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        digits=4,
        zero_division=0,
    )
    with (OUTPUT_DIR / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report)

    matrix = confusion_matrix(test_labels, predicted_labels, labels=range(len(CLASSES)))
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predicted letter")
    plt.ylabel("True letter")
    plt.title("SignSpeak test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with (OUTPUT_DIR / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2)
    save_training_plots(history_data)

    best_epoch = int(np.argmax(history_data["val_accuracy"]))
    metrics = {
        "training_accuracy": history_data["accuracy"][best_epoch],
        "validation_accuracy": history_data["val_accuracy"][best_epoch],
        "testing_accuracy": float(test_accuracy),
        "testing_loss": float(test_loss),
        "number_of_classes": len(CLASSES),
        "training_examples": int(len(train_images)),
        "test_examples": int(len(test_images)),
        "epochs_completed": len(history_data["accuracy"]),
    }
    with (OUTPUT_DIR / "evaluation_results.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    mapping = {
        "classes": list(CLASSES),
        "dataset_labels": list(DATASET_LABELS),
        "dataset_label_to_character": {
            str(label): DATASET_LABEL_TO_CLASS[label] for label in DATASET_LABELS
        },
        "dataset_label_to_model_index": {
            str(label): index for index, label in enumerate(DATASET_LABELS)
        },
    }
    with (MODEL_DIR / "class_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)

    with (OUTPUT_DIR / "model_summary.txt").open("w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))
    print(json.dumps(metrics, indent=2))
    print(f"Saved trained model: {checkpoint_path}")
    print(f"Saved evaluation artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
