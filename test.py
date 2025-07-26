import cv2
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
import os

# --- Custom Loss Function: Wing Loss ---
def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    IMG_HEIGHT = 128  # Must match training
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

# --- Configuration ---
MODEL_PATH = Path("saved_models_v2/pupil_model_best.keras")
IMG_HEIGHT = 128
IMG_WIDTH = 128
OUTPUT_DIR = Path("output_frames")

# Try to use imshow, fallback to saving if not available
USE_IMSHOW = True
try:
    cv2.namedWindow("TestWindow")
    cv2.destroyWindow("TestWindow")
except Exception:
    USE_IMSHOW = False
    print("cv2.imshow is not available. Output will be saved to disk.")


def predict_pupil_coordinates_from_frame(frame, model):
    original_height, original_width = frame.shape[:2]
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (IMG_WIDTH, IMG_HEIGHT))
    normalized_img = resized_img.astype(np.float32) / 255.0
    img_3ch = np.stack([normalized_img]*3, axis=-1)  # (H, W, 3)
    input_tensor = np.expand_dims(img_3ch, axis=0)   # (1, H, W, 3)
    normalized_coords = model.predict(input_tensor)
    norm_x, norm_y = normalized_coords[0]
    pred_x = int(norm_x * original_width)
    pred_y = int(norm_y * original_height)
    return (pred_x, pred_y)

def process_image(image_path, model):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    coords = predict_pupil_coordinates_from_frame(img, model)
    # Draw prediction
    cv2.circle(img, coords, 5, (0,255,0), -1)
    cv2.circle(img, coords, 7, (0,0,255), 2)
    print(f"Predicted Coordinates (x, y): {coords}")
    if USE_IMSHOW:
        cv2.imshow("Pupil Prediction", img)
        print("Press any key to close the image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"predicted_{image_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"Output saved to {out_path}")

def process_video(video_path, model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frame_idx = 0
    OUTPUT_DIR.mkdir(exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        coords = predict_pupil_coordinates_from_frame(frame, model)
        # Draw prediction
        cv2.circle(frame, coords, 5, (0,255,0), -1)
        cv2.circle(frame, coords, 7, (0,0,255), 2)
        if USE_IMSHOW:
            cv2.imshow("Pupil Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out_path = OUTPUT_DIR / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
        frame_idx += 1
    cap.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
    else:
        print(f"All output frames saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pupil location in an eye image or video.")
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Path to the input image or video file (e.g., test_images/sample_eye.jpg or test_images/sample_video.mp4)"
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)

    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train.py first to train and save the model.")
    elif not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
    else:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'wing_loss': wing_loss})
        # Determine if input is image or video by extension
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            process_image(input_path, model)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            process_video(input_path, model)
        else:
            print(f"Unsupported file type: {input_path.suffix}")