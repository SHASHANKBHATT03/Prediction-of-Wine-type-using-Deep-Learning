# Prediction-of-Wine-type-using-Deep-Learning

**Project Title**: Wine Type Prediction using Deep Learning 

**Overview**:
This project implements a deep learning model to classify wine samples into three categories (Class 0, 1, or 2) based on their physicochemical attributes. The model is trained using the UCI Wine Dataset and implemented in Python using TensorFlow/Keras.

---

**Dataset**:

* **Source**: UCI Machine Learning Repository (Wine Dataset)
* **Features**: 13 numerical features describing chemical properties of wine.
* **Target**: Wine class (0, 1, or 2)

---

**Technologies Used**:

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib (for visualization)

---

**Steps Followed**:

1. **Data Loading** – Loaded dataset using `sklearn.datasets.load_wine()`.
2. **Preprocessing** – Applied normalization using `StandardScaler`.
3. **Train-Test Split** – 80% training, 20% testing using `train_test_split`.
4. **Model Building** – Created a Sequential deep learning model with:

   * Input layer of 13 neurons
   * Hidden layers with ReLU activation
   * Output layer with 3 neurons (softmax for multi-class classification)
5. **Training** – Model trained for **10 epochs**.
6. **Evaluation** – Accuracy calculated on test data and plotted using a confusion matrix.

---

**Model Configuration**:

* Loss Function: `sparse_categorical_crossentropy`
* Optimizer: `adam`
* Metrics: `accuracy`
* Epochs: **10**
* Batch Size: default

---

**Results**:

* Training Accuracy: \[your result here]
* Test Accuracy: \[your result here]
* Confusion Matrix: \[brief note if plotted]

---

**How to Run**:

1. Install required libraries:

   ```bash
   pip install numpy pandas scikit-learn tensorflow matplotlib
   ```
2. Run the Python file:

   ```bash
   python wine_prediction.py
   ```

---

**Note**: This implementation uses only 10 epochs for faster training and experimentation. Accuracy might vary slightly compared to higher epoch runs.


