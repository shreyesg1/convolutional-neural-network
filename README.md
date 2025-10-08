# Convolutional Neural Network: NumPy Deep Learning Framework

A complete deep learning framework built from scratch using only **NumPy**, featuring comprehensive mathematical documentation and achieving **97.70% test accuracy** on MNIST.

This project was done over the course of two months. The majority of the time spent was on learning the theory behind machine learning and applying it mathematically in code. The goal of the project was to develop a CNN that scored higher on the MNIST than both the TensorFlow and PyTorch libraries in less time. This is made possible due to these libraries being generalized for commercial use, while this project is hyperoptimized for one specific architecture and can cut corners to save time (but not optimized *purely* for MNIST; it still delivers good results on other datasets, such as Fashion MNIST). At the moment, layers like Dense and Softmax achieve, respectively, 50x and 150x faster runtimes than those of popular libraries. However, the main time losses are found in the backward pass of the training loop, which libraries often skip due to more optimal alternatives. As this project is further developed, a similar approach will be taken.

All code seen in this project is handwritten and was done with assistance from only numpy documentation, research papers, and web articles. There was zero Generative AI use to write any code, and any AI use was restricted to file headers and the remainder of this README file. To clarify, all comments left throughout the files that are not part of a function header were also handwritten as a way for me to articulate my thoughts and solidify the mathematical intuition.

 ---

## Key Features

* **Pure NumPy implementation** – No external deep learning libraries
* **Mathematical documentation** – Every operation explained with formulas and derivations
* **Efficient algorithms** – im2col convolution, Adam optimizer, structured matrices

---

## Architecture

```
Conv2D(8, 3×3) → ReLU → MaxPool(2×2)  
→ Conv2D(16, 3×3) → ReLU → MaxPool(3×3)  
→ Flatten → Dense(256) → ReLU → Dropout(0.25)  
→ Dense(256) → Dense(10)
```

---

## Results

* **Training Accuracy:** 98.08%
* **Test Accuracy:** 97.70%
* **Parameters:** ~50K (efficient architecture)
* **Training Time:** 20 epochs, batch size 512

---

## Mathematical Highlights

* **Convolution:** im2col transformation reducing convolution complexity from *O(n²)* → *O(n)*
* **Adam Optimizer:** Bias-corrected moment estimates with adaptive learning rates
* **Structured Matrices:** Walsh-Hadamard transforms reducing parameters by ~90%
* **Numerical Stability:** log-sum-exp trick, gradient clipping, proper initialization

---

## Quick Start

Install dependencies (note that TensorFlow and Keras are only utilized in the MNIST initialization, while tqdm is purely for aesthetic purposes):

```bash
pip install numpy tensorflow keras tqdm
```

Train the model:

```bash
python main.py
```

---

## Project Structure

```plaintext
layers/      - Neural network components (Conv2D, Dense, ReLU, etc.)
utilities/   - Optimizers, loss functions, metrics
model.py     - Model container and training pipeline
main.py      - MNIST training script
```

---
