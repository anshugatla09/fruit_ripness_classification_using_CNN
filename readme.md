# Fruit Ripeness Classification Using CNNs

## Project Overview

This project presents a deep learning approach for fruit ripeness and freshness classification using Convolutional Neural Networks (CNNs). The model classifies fruit images into six categories:

- Fresh Apples
- Fresh Bananas
- Fresh Oranges
- Rotten Apples
- Rotten Bananas
- Rotten Oranges

The objective is to automate fruit quality assessment using image classification techniques, reducing the need for manual inspection.

---

## Dataset

Dataset Source:

https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

### Dataset Classes

- freshapples
- freshbanana
- freshoranges
- rottenapples
- rottenbanana
- rottenoranges

### Dataset Structure

dataset/
├── train/
│   ├── freshapples
│   ├── freshbanana
│   ├── freshoranges
│   ├── rottenapples
│   ├── rottenbanana
│   └── rottenoranges
│
└── test/
    ├── freshapples
    ├── freshbanana
    ├── freshoranges
    ├── rottenapples
    ├── rottenbanana
    └── rottenoranges

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Notebook

---

## Methodology

### 1. Data Preprocessing

- Image resizing (224 × 224)
- Pixel normalization
- Data augmentation:
  - Rotation
  - Horizontal Flip
  - Zoom

### 2. Custom CNN Model

Architecture:

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Conv2D (128 Filters)
- MaxPooling2D
- Flatten Layer
- Dense Layer (256 Neurons)
- Dropout Layer
- Output Layer (6 Classes)

### 3. Transfer Learning Model

MobileNetV2 was used as the transfer learning model for performance comparison.

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Results

### Custom CNN

Accuracy: 97.59%

### MobileNetV2

Accuracy: 98.93%

### Model Comparison

| Model | Accuracy |
|---------|----------|
| Custom CNN | 97.59% |
| MobileNetV2 | 98.93% |

MobileNetV2 achieved higher accuracy due to the advantages of transfer learning and pre-trained feature extraction.

---

## Output Visualizations

The project generates:

- Accuracy Curve
- Loss Curve
- Confusion Matrix
- Classification Report
- Model Comparison Results

---

## Project Structure

Fruit-Ripeness-Classification-Using-CNN/

├── README.md

├── ml-cnn-model.ipynb

├── accuracy_curve.png

├── loss_curve.png

├── confusion_matrix.png

├── cnn_model.h5

├── mobilenet_model.h5

└── outputs/

---

## How to Run

### 1. Clone Repository

```bash
git clone <repository-link>
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 3. Download Dataset

Download from:

https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

### 4. Update Dataset Path

Modify:

```python
train_dir = "dataset/train"
test_dir = "dataset/test"
```

according to your local dataset location.

### 5. Run Notebook

```bash
jupyter notebook
```

Open:

```text
ml-cnn-model.ipynb
```

and run all cells.

---

## Conclusion

A CNN-based fruit ripeness classification system was successfully developed and evaluated using a publicly available fruit dataset. The custom CNN model achieved high classification accuracy, while MobileNetV2 further improved performance through transfer learning. The results demonstrate the effectiveness of deep learning techniques for automated fruit quality assessment in agricultural applications.

---

## Author

Anshu Kumar

Machine Learning Project Phase II

Fruit Ripeness Classification Using CNNs