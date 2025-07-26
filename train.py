# ==============================================================================
#  High-Accuracy Pupil Tracker using Transfer Learning and Advanced Techniques
# ==============================================================================

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import datetime
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint

# --- Configuration Parameters ---
DATA_DIR = Path("data/lpw")
SAVE_DIR = Path("saved_models_v2")
LOG_DIR = Path("logs/fit_v2")
IMG_HEIGHT = 128  # Adjusted for MobileNetV2 efficiency
IMG_WIDTH = 128
FRAME_SAMPLING_INTERVAL = 10 # Sample more frames for a larger dataset
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# --- Custom Loss Function: Wing Loss ---
def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    """
    Custom Wing Loss function for Keras/TensorFlow.
    Reference: https://arxiv.org/abs/1711.06753
    The loss is calculated on the normalized coordinates (0-1 range).
    To make the loss meaningful, we scale the error to pixel space.
    """
    y_true_px = y_true * IMG_HEIGHT
    y_pred_px = y_pred * IMG_HEIGHT
    
    x = y_true_px - y_pred_px
    abs_x = tf.abs(x)
    
    C = w - w * tf.math.log(1.0 + w / epsilon)
    
    nonlinear_part = w * tf.math.log(1.0 + abs_x / epsilon)
    linear_part = abs_x - C
    
    is_small_error = tf.cast(abs_x < w, tf.float32)
    
    loss = is_small_error * nonlinear_part + (1.0 - is_small_error) * linear_part
    return tf.reduce_mean(loss)

# --- Data Augmentation Pipeline using imgaug ---
# Define separate pipelines for training (heavy augmentation) and validation (only resizing)
train_augmenter = iaa.Sequential([
    iaa.Resize({"height": IMG_HEIGHT, "width": IMG_WIDTH}),
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-10, 10),
        shear=(-5, 5),
        mode="edge"
    ),
    iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
    iaa.GammaContrast((0.8, 1.2)),
], random_order=True)

val_augmenter = iaa.Sequential([
    iaa.Resize({"height": IMG_HEIGHT, "width": IMG_WIDTH})
])

# --- Data Generator Class ---
class PupilDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, annotations, batch_size, augmenter, shuffle=True):
        self.image_paths = image_paths
        self.annotations = annotations
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indexes]
        batch_annotations = [self.annotations[i] for i in batch_indexes]
        X, y = self.__data_generation(batch_image_paths, batch_annotations)
        return X, y

    def __data_generation(self, batch_paths, batch_annos):
        batch_images = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        batch_keypoints = np.zeros((self.batch_size, 2), dtype=np.float32)

        for i, (path, anno) in enumerate(zip(batch_paths, batch_annos)):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            
            image = self.clahe.apply(image)

            keypoints_on_image = KeypointsOnImage([Keypoint(x=anno[0], y=anno[1])], shape=image.shape)
            
            img_aug, kps_aug = self.augmenter(image=image, keypoints=keypoints_on_image)

            img_aug_normalized = img_aug.astype(np.float32) / 255.0
            img_aug_normalized = np.stack([img_aug_normalized]*3, axis=-1)  # Convert to (H, W, 3)
            
            # Extract augmented keypoints
            aug_x = kps_aug.keypoints[0].x
            aug_y = kps_aug.keypoints[0].y
            
            # Normalize coordinates to  range and clip to handle out-of-bounds keypoints
            norm_x = np.clip(aug_x / IMG_WIDTH, 0.0, 1.0)
            norm_y = np.clip(aug_y / IMG_HEIGHT, 0.0, 1.0)

            batch_images[i] = img_aug_normalized
            batch_keypoints[i] = [norm_x, norm_y]
            
        return batch_images, batch_keypoints

# --- Data Loading Function ---
def load_paths_and_annotations(data_path: Path):
    video_dir = data_path / "videos"
    annotation_dir = data_path / "annotations"
    video_files = sorted(list(video_dir.glob("*.avi")))
    
    all_frame_paths = []
    all_annotations = []
    
    # Create a directory to store extracted frames to speed up subsequent runs
    frames_dir = data_path / "extracted_frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"Found {len(video_files)} video files. Starting frame extraction and annotation loading...")
    for video_path in tqdm(video_files, desc="Processing Videos"):
        annotation_file = annotation_dir / f"{video_path.stem}.txt"
        if not annotation_file.exists(): continue
            
        annotations_df = pd.read_csv(annotation_file, header=None, sep=' ', names=['x_coord', 'y_coord'])
        cap = cv2.VideoCapture(str(video_path))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % FRAME_SAMPLING_INTERVAL == 0 and frame_idx < len(annotations_df):
                frame_save_path = frames_dir / f"{video_path.stem}_frame_{frame_idx}.png"
                
                if not frame_save_path.exists():
                    cv2.imwrite(str(frame_save_path), frame)

                all_frame_paths.append(frame_save_path)
                coords = annotations_df.iloc[frame_idx]
                # Store original coordinates, normalization will happen in the generator
                all_annotations.append([coords['x_coord'], coords['y_coord']])
            
            frame_idx += 1
        cap.release()
        
    return all_frame_paths, all_annotations

# --- Model Building Function (Transfer Learning) ---
def build_transfer_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = layers.Input(shape=input_shape)
    
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        weights='imagenet',
        include_top=False,
        alpha=0.75 # Using a smaller variant for efficiency
    )
    backbone.trainable = False # Freeze the backbone

    # Custom regression head
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = layers.Dense(2, activation='sigmoid', name='pupil_coords')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PupilTracker_MobileNetV2")
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load data paths and annotations
    all_paths, all_annos = load_paths_and_annotations(DATA_DIR)
    print(f"Data loaded successfully. Found {len(all_paths)} frames.")

    # Split data into training and validation sets
    paths_train, paths_val, annos_train, annos_val = train_test_split(
        all_paths, all_annos, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    print(f"Training set size: {len(paths_train)}, Validation set size: {len(paths_val)}")

    # Create data generators
    train_generator = PupilDataGenerator(paths_train, annos_train, BATCH_SIZE, train_augmenter)
    val_generator = PupilDataGenerator(paths_val, annos_val, BATCH_SIZE, val_augmenter, shuffle=False)

    # Build and compile the model
    model = build_transfer_model()
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=wing_loss, metrics=['mae'])
    model.summary()

    # Setup callbacks
    SAVE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=str(SAVE_DIR / 'pupil_model_best.keras'),
        save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1
    )
    tensorboard_callback = TensorBoard(
        log_dir=str(LOG_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    # --- Start Model Training ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[model_checkpoint_callback, early_stopping, lr_scheduler, tensorboard_callback]
    )
    print("--- Model Training Finished ---")
    print(f"Best model saved to {SAVE_DIR / 'pupil_model_best.keras'}")