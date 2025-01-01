# Pure NumPy Neural Network 🌟

This project implements a **neural network from scratch** using only **NumPy**, based on the tutorial by **Samson Zhang** on YouTube. The network demonstrates how to handle forward propagation, backpropagation, and weight updates without relying on deep learning libraries like TensorFlow or PyTorch.

---

## Features 🚀

- **No External Frameworks**: Entirely built using NumPy for matrix operations.
- **Custom Training Pipeline**: Implements forward propagation, loss calculation, backpropagation, and gradient descent manually.
- **End-to-End Learning**: Includes data preprocessing, model training, and evaluation.

---

## How It Works 🧠

### 1. **Data Preprocessing**
   - The input data is normalized by dividing pixel values by `255` to scale them into the range `[0, 1]`.
   - Labels were one hot encoded.
### 2. **Forward Propagation**
   - Compute activations for each layer using weighted sums and activation functions.
   - Activations pass through non-linearities using softmax and ReLU for more expressive power.

### 3. **Loss Calculation**
   - Compute the error between predictions and true labels.

### 4. **Backpropagation**
   - Derive gradients for weights and biases by manually implementing the chain rule.
   - Propagate errors backward through the network to update parameters.

### 5. **Gradient Descent**
   - Use gradients to adjust weights and biases, minimizing the loss over multiple iterations (epochs).

---

## Example Usage 📖

### Training the Model
1. Utilized the popular MNIST dataset:
   - Ensure input features are normalized (e.g., pixel values scaled to `[0, 1]`).
  

2. Train the neural network:
   - Run the forward and backward passes for a set number of epochs.
   - Adjust weights using gradient descent.

3. Evaluate the model:
   - Test the trained network on unseen data to measure accuracy and generalization. (85% Accuracy).

## Final notes 
This project was inspired by the tutorial done by Samson Zhang please check it out https://www.youtube.com/watch?v=w8yWXqWQYmU!
This tutorial was followed to soldify my technical skills in applying theoritical cornerstones of machine learning such as backpropagation,forward propagation and updating the weights & biases according to their contribution to the error.
