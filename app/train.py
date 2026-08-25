"""Train the solar panel fault-detection CNN and save it for serving.

Adapted from the original SolarPV_CNN_project.py: same dataset layout,
same architecture, but headless (no interactive plotting) so it can run
in a container or CI job.
"""
import os
import shutil
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATASET_DIR = os.path.join(BASE, "data", "hf_download", "Faulty_solar_panel_Train")
PROCESSED_DIR = os.path.join(BASE, "data", "Processed_Dataset")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
CLEAN_DIR = os.path.join(ORIGINAL_DATASET_DIR, "Clean")
FAULT_CLASSES = ["Bird-drop", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
MODEL_DIR = os.path.join(BASE, "app", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "solar_fault_classifier.h5")


def prepare_folders():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(os.path.join(TRAIN_DIR, "Clean"))
    os.makedirs(os.path.join(TRAIN_DIR, "Defect"))
    os.makedirs(os.path.join(VAL_DIR, "Clean"))
    os.makedirs(os.path.join(VAL_DIR, "Defect"))


def split_images(source_dir, train_dst, val_dst, split_ratio=0.8):
    images = os.listdir(source_dir)
    train_imgs, val_imgs = train_test_split(images, train_size=split_ratio, random_state=42)
    for img in train_imgs:
        shutil.copy(os.path.join(source_dir, img), os.path.join(train_dst, img))
    for img in val_imgs:
        shutil.copy(os.path.join(source_dir, img), os.path.join(val_dst, img))


def prepare_dataset():
    split_images(CLEAN_DIR, os.path.join(TRAIN_DIR, "Clean"), os.path.join(VAL_DIR, "Clean"))
    defect_images = []
    for folder in FAULT_CLASSES:
        folder_path = os.path.join(ORIGINAL_DATASET_DIR, folder)
        if os.path.exists(folder_path):
            defect_images += [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
    random.shuffle(defect_images)
    split = int(0.8 * len(defect_images))
    for img in defect_images[:split]:
        shutil.copy(img, os.path.join(TRAIN_DIR, "Defect"))
    for img in defect_images[split:]:
        shutil.copy(img, os.path.join(VAL_DIR, "Defect"))


def load_data():
    train_gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_gen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=32, class_mode="binary")
    val = val_gen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=32, class_mode="binary")
    return train, val


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_data, val_data, epochs=10):
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return history


if __name__ == "__main__":
    print("Starting Solar PV Fault Detection training")

    if not os.path.exists(ORIGINAL_DATASET_DIR):
        raise SystemExit(f"Dataset folder '{ORIGINAL_DATASET_DIR}' not found!")

    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        print("Dataset already prepared. Skipping data prep.")
    else:
        print("Preparing dataset...")
        prepare_folders()
        prepare_dataset()

    print("Loading image data...")
    train_data, val_data = load_data()

    print("Building and training model...")
    model = build_model()
    train_model(model, train_data, val_data)
