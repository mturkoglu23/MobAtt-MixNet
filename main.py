# ===============================
# Fish Image Classification with Backbone + Attention-Enhanced MLP-Mixer
# Author: [Muammer Turkoglu et al.]
# ===============================

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATASET_DIR = 'dataset_new/fish'
IMG_SIZE = 224
NUM_CLASSES = 5
NUM_FOLDS = 5
BATCH_SIZE = 16
NUM_EPOCHS = 100
SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2
MOMENTUM = 0.2
BACKBONE_MODEL = "mobilenetv2"
OPTIMIZER_CHOICE = "AdamW"

# === DATA PREPARATION ===
def load_dataset(dataset_dir, img_size):
    labels, images = [], []
    class_dirs = os.listdir(dataset_dir)

    for label in class_dirs:
        label_dir = os.path.join(dataset_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return images, labels_encoded, label_encoder.classes_

# === MODEL COMPONENTS ===
def get_backbone(backbone_name, input_shape):
    backbones = {
        "mobilenetv2": keras.applications.MobileNet(
            include_top=False, weights="imagenet", input_shape=input_shape
        ),
    }
    return backbones[backbone_name]

def build_feature_extractor(backbone, optimizer):
    x = layers.GlobalAveragePooling2D()(backbone.output)
    model = models.Model(inputs=backbone.input, outputs=x)

    for layer in model.layers:
        layer.trainable = False

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    return model

def attention_block(x):
    attention_scores = layers.Attention()([x, x])
    x = layers.Add()([x, attention_scores])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def mlp_block(x, hidden_dim, mlp_ratio=4.0):
    channels = x.shape[-1]
    mlp_hidden_dim = int(channels * mlp_ratio)
    x = layers.Dense(mlp_hidden_dim, activation='gelu')(x)
    x = layers.Dense(hidden_dim)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return x

def mixer_block(x, hidden_dim, mlp_ratio=4.0):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    x_att = attention_block(x1)
    x_mlp = mlp_block(x_att, hidden_dim, mlp_ratio)
    x2 = layers.Add()([x_att, x_mlp, x])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x_att1 = attention_block(x3)
    x_mlp = mlp_block(x_att1, hidden_dim, mlp_ratio)
    x4 = layers.Add()([x_att1, x_mlp, x2])

    return layers.LayerNormalization(epsilon=1e-6)(x4)

def build_mlp_mixer(input_shape, num_classes, num_blocks, hidden_dim):
    inputs = layers.Input(shape=input_shape)
    split_features = tf.split(inputs, num_blocks, axis=1)

    for i in range(num_blocks):
        split_features[i] = mixer_block(split_features[i], hidden_dim)

    features = tf.concat(split_features, axis=1)
    outputs = layers.Dense(num_classes, activation='softmax')(features)

    return models.Model(inputs, outputs, name="mlp_mixer")

# === OPTIMIZERS ===
def get_optimizer(name):
    optimizers = {
        "AdamW": tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        "SGD": keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
        "Adam": keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    }
    return optimizers[name]

# === MAIN TRAINING LOOP ===
def main():
    # Load and preprocess dataset
    images, labels, class_names = load_dataset(DATASET_DIR, IMG_SIZE)
    print(f"Loaded {len(images)} images across {len(class_names)} classes.")

    # Initialize backbone and feature extractor
    backbone = get_backbone(BACKBONE_MODEL, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    optimizer = get_optimizer(OPTIMIZER_CHOICE)
    feature_extractor = build_feature_extractor(backbone, optimizer)

    # Extract features
    features = feature_extractor.predict(images, batch_size=BATCH_SIZE)
    print("Extracted features shape:", features.shape)

    hidden_dim = features.shape[-1] // 4
    num_blocks = 4

    accuracies = []
    overall_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        print(f"\n=== Fold {fold + 1}/{NUM_FOLDS} ===")

        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Build and compile model
        model = build_mlp_mixer(
            input_shape=(x_train.shape[1],),
            num_classes=NUM_CLASSES,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim
        )
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.TopKCategoricalAccuracy(5, name="top5-acc"),
            ],
        )

        # Train
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )

        # Evaluate
        y_pred = model.predict(x_val, batch_size=BATCH_SIZE)
        y_pred_classes = np.argmax(y_pred, axis=1)
        fold_accuracy = accuracy_score(y_val, y_pred_classes)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

        cm = confusion_matrix(y_val, y_pred_classes)
        overall_cm += cm
        accuracies.append(fold_accuracy)

    print("\n=== FINAL RESULTS ===")
    print(f"Average Accuracy across {NUM_FOLDS} folds: {np.mean(accuracies):.4f}")
    print("Overall Confusion Matrix:\n", overall_cm)

if __name__ == "__main__":
    main()
