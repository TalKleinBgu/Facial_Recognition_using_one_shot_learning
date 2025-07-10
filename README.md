# Deep Learning HW2 – Face Verification Using Siamese Networks


## 🧠 Project Overview

This project focuses on training a **Siamese Neural Network** on the [LFW-a Dataset](http://vis-www.cs.umass.edu/lfw/) to verify whether two face images belong to the same person. The network learns a similarity function rather than a direct classification.
> 📄 Our work is based on the paper:  
> **[Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)** by Gregory Koch et al.  
> This architecture forms the foundation of our approach to comparing face embeddings.
---

## 📁 Dataset Summary

- **Dataset**: Labeled Faces in the Wild-a (LFW-a)
- **Total People**: 5749  
- **Total Images**: 13,233 (Grayscale)  
- **Train Pairs**: 2200  
- **Test Pairs**: 1000  
- **Split**: 80/20 Train/Val from Train Set

> 📷 Example Images:
> 
> ![](images/image1.png)

---

## 🧱 Model Architecture

The model is a **Siamese CNN** consisting of:

- 8 convolutional layers  
- `ReLU` activations  
- 3 × MaxPooling  
- 1 × Fully Connected layer  
- Adapted kernel sizes: `25×25`, `18×18`, `10×10` (due to input size 250×250)

> 🧩 Architecture Illustration by the paper:  
> ![](images/image2.png)

---

## ⚙️ Training Configuration

- **Batch Sizes**: 32, 64  
- **Optimizers**: SGD, Adam  
- **Loss Function**: `Binary Cross Entropy`  
- **Learning Rate**: 0.01  
- **Scheduler**: `StepLR`  
- **Epochs**: Up to 15 (with early stopping)  
- **Dropout**: Used in some experiments (rate = 0.4)

---

## 🔬 Experiments Summary

The following table summarizes all six experiments, including optimizer, batch size, use of batch normalization and dropout, and the results:

| Exp | Epochs | Batch Size | Optimizer | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Test Loss | Test Accuracy | Time Taken | BatchNorm | Dropout |
|-----|--------|------------|-----------|-------------|----------------|-----------|---------------|------------|----------------|-------------|------------|---------|
| 1   | 14     | 32         | SGD       | 0.506       | 80.11%         | 0.598     | 75.42%        | 0.663      | 68.7%          | 104.835     | ❌         | ❌      |
| 2   | 7      | 32         | SGD       | 0.406       | **85.52%**     | 0.675     | 73.58%        | 0.667      | **72.9%**      | 47.134      | ✅         | ❌      |
| 3   | 10     | 64         | SGD       | 0.294       | 90.49%         | 0.600     | 75.57%        | 0.597      | 72.0%          | 50.642      | ✅         | ❌      |
| 4   | 12     | 32         | Adam      | 0.695       | 65.51%         | 0.733     | 65.97%        | 0.688      | 54.2%          | 52.579      | ✅         | ❌      |
| 5   | 6      | 64         | Adam      | 0.700       | 63.24%         | 0.708     | 61.11%        | 0.725      | 52.2%          | 19.484      | ✅         | ❌      |
| 6   | 7      | 32         | SGD       | 0.620       | 72.47%         | 0.715     | 62.50%        | 0.676      | 59.0%          | 18.307      | ✅         | ✅      |

- ✅ = Used in experiment  
- ❌ = Not used  
- **Bold** = Best performance for that metric


---

## 📊 Key Insights

- **BatchNorm**: Boosted accuracy and convergence (especially Exp 2 vs. Exp 1)
- **SGD vs. Adam**: SGD outperformed Adam in both convergence and accuracy
- **Dropout**: Reduced overfitting slightly (Exp 6), but aggressive dropout can hurt
- **Batch Size**: Smaller batch size helped generalization (32 > 64)
- **Kernel Sizes**: Larger receptive fields improved performance for 250×250 input

---

## ✅ Prediction Examples

> Correct (True Positive):  
> ![](images/image3.png)

> Correct (True Negative):  
> ![](images/image4.png)

> ❌ Incorrect (False Positive):  
> ![](images/image5.png)

> ❌ Incorrect (False Negative):  
> ![](images/image6.png)

---
