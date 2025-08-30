
# 🖼️ Image Classification with TensorFlow & Keras

This project demonstrates how to build a simple **image classification model** using **TensorFlow** and **Keras**.  
We use a neural network with fully connected (Dense) layers to classify images from the **MNIST dataset** (handwritten digits 0–9).

---

## 🚀 Project Overview
- Dataset: **MNIST** (60,000 training + 10,000 test grayscale images, 28×28 pixels)
- Model type: **Sequential Neural Network**
- Goal: Predict which digit (0–9) appears in an input image.

---

## 🏗️ Model Architecture

```text
Input (28x28 image)
   ↓
Flatten (28x28 → 784)
   ↓
Dense(128, activation='relu')
   ↓
Dense(10, output logits for classes 0–9)
````

* **Flatten**: Converts 2D image into 1D vector.
* **Dense(128, relu)**: Hidden layer that learns features.
* **Dense(10)**: Output layer for classification (10 digits).

---

## ⚙️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Open the notebook:

   ```bash
   jupyter notebook Image_Classification_With_Tensorflow.ipynb
   ```

---

## 🏃 Running the Model

Inside the notebook:

1. Load the MNIST dataset.
2. Preprocess images (normalize pixel values).
3. Define and compile the Sequential model.
4. Train the model using:

   ```python
   model.fit(x_train, y_train, epochs=5)
   ```
5. Evaluate on test data:

   ```python
   model.evaluate(x_test, y_test)
   ```

---

## 📊 Results

* Achieves **\~97–98% accuracy** on the MNIST test set after training for 5 epochs.
* Example prediction:

  ```python
  predictions = model.predict(x_test[:5])
  ```

---

## 📌 Future Improvements

* Add **Convolutional Neural Networks (CNNs)** for higher accuracy.
* Experiment with **Dropout layers** to reduce overfitting.
* Try on other datasets (e.g., Fashion-MNIST, CIFAR-10).
