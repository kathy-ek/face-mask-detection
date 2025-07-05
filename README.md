# Face Mask Detection with TensorFlow and OpenCV

This project implements a real-time face mask detection system using a pre-trained TensorFlow/Keras model and OpenCV for video capture and face detection.

## 📁 Project Structure

```
project-root/
│
├── dataset/               # Original dataset with 'with_mask' and 'without_mask' folders
├── training/              # Training dataset (split by script)
├── validation/            # Validation dataset (split by script)
├── my_model.h5            # Trained Keras model file
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── mask_detection.py      # Main Python script
└── README.md              # This file
```

## 🚀 Features

- Splits original dataset into training and validation sets (80/20 split).
- Uses a trained CNN model to classify faces with or without masks in real-time.
- Detects faces using OpenCV's Haar Cascade classifier.
- Displays webcam feed with bounding boxes and labels ("Mask" or "No Mask").

## 🛠️ Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

Install all dependencies with:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## 📦 Dataset Preparation

1. Organize your dataset into the following structure:

```
dataset/
├── with_mask/
├── without_mask/
```

2. The script will automatically create and populate:

```
training/
├── with_mask/
├── without_mask/

validation/
├── with_mask/
├── without_mask/
```

using an 80/20 training-validation split.

## 🧠 Model

The model should be a Keras `.h5` file trained on the dataset.  
Requirements:
- Input shape: `(128, 128, 3)`
- Output: Sigmoid activation for binary classification (mask vs. no mask)

**Note**: This script assumes the model is saved as `my_model.h5` in the root directory.

## 🎥 Running Real-Time Detection

1. Make sure `my_model.h5` and `haarcascade_frontalface_default.xml` are in the project folder.
2. Run the script:

```bash
python mask_detection.py
```

3. A webcam window will open and start detecting masks in real time.

- Green label: `Mask`
- Red label: `No Mask`
- Press `q` to exit the webcam window.

## 🧪 Face Detection and Prediction

- Uses OpenCV Haar Cascade to detect faces.
- Extracts each face, resizes to `128x128`, normalizes pixel values.
- Passes it to the model for prediction.
- Draws a rectangle and label based on prediction confidence.

```python
threshold = 0.4
if prediction > threshold:
    label = 'Mask'
else:
    label = 'No Mask'
```

Adjust `threshold` based on your model performance.

## ⚠️ Notes

- Check the path to `haarcascade_frontalface_default.xml` and `my_model.h5`
- Webcam should be connected and working
- Model must be trained separately before using this script

## 📬 Contact

For questions or contributions, feel free to reach out or open an issue on the repository.
# Face Mask Detection with TensorFlow and OpenCV

This project implements a real-time face mask detection system using a pre-trained TensorFlow/Keras model and OpenCV for video capture and face detection.

## 📁 Project Structure

```
project-root/
│
├── dataset/               # Original dataset with 'with_mask' and 'without_mask' folders
├── training/              # Training dataset (split by script)
├── validation/            # Validation dataset (split by script)
├── my_model.h5            # Trained Keras model file
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── mask_detection.py      # Main Python script
└── README.md              # This file
```

## 🚀 Features

- Splits original dataset into training and validation sets (80/20 split).
- Uses a trained CNN model to classify faces with or without masks in real-time.
- Detects faces using OpenCV's Haar Cascade classifier.
- Displays webcam feed with bounding boxes and labels ("Mask" or "No Mask").

## 🛠️ Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

Install all dependencies with:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## 📦 Dataset Preparation

1. Organize your dataset into the following structure:

```
dataset/
├── with_mask/
├── without_mask/
```

2. The script will automatically create and populate:

```
training/
├── with_mask/
├── without_mask/

validation/
├── with_mask/
├── without_mask/
```

using an 80/20 training-validation split.

## 🧠 Model

The model should be a Keras `.h5` file trained on the dataset.  
Requirements:
- Input shape: `(128, 128, 3)`
- Output: Sigmoid activation for binary classification (mask vs. no mask)

**Note**: This script assumes the model is saved as `my_model.h5` in the root directory.

## 🎥 Running Real-Time Detection

1. Make sure `my_model.h5` and `haarcascade_frontalface_default.xml` are in the project folder.
2. Run the script:

```bash
python mask_detection.py
```

3. A webcam window will open and start detecting masks in real time.

- Green label: `Mask`
- Red label: `No Mask`
- Press `q` to exit the webcam window.

## 🧪 Face Detection and Prediction

- Uses OpenCV Haar Cascade to detect faces.
- Extracts each face, resizes to `128x128`, normalizes pixel values.
- Passes it to the model for prediction.
- Draws a rectangle and label based on prediction confidence.

```python
threshold = 0.4
if prediction > threshold:
    label = 'Mask'
else:
    label = 'No Mask'
```

Adjust `threshold` based on your model performance.

## ⚠️ Notes

- Check the path to `haarcascade_frontalface_default.xml` and `my_model.h5`
- Webcam should be connected and working
- Model must be trained separately before using this script

## 📬 Contact

For questions or contributions, feel free to reach out or open an issue on the repository.
