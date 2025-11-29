# Neural Network Math

This document explains the neural network at the heart of the demo, focusing on the **2 → 4 → 8 → 2** multilayer perceptron implemented in `ToyNet`. The goal is to connect the **equations** you see in textbooks to the **actual C++ code**.

The relevant files are:

- `include/core/ToyNet.h`
- `src/core/ToyNet.cpp`
- `include/core/Trainer.h`
- `src/core/Trainer.cpp`

---

## 1. Architecture

The network is a small fully-connected classifier:

- **Inputs**: 2D point `(x, y)` in normalized device coordinates.
- **Hidden layer 1**: 4 units, ReLU activation.
- **Hidden layer 2**: 8 units, ReLU activation.
- **Output layer**: 2 units, softmax activation (class probabilities).

In code (see `ToyNet.h`):

```cpp
static constexpr int InputDim  = 2;
static constexpr int Hidden1   = 4;
static constexpr int Hidden2   = 8;
static constexpr int OutputDim = 2;
```

We treat the network as mapping:

```text
R^2  →  R^4  →  R^8  →  R^2
(x,y)   h1       h2      logits → softmax → probabilities
```

---

## 2. Parameters and notation

For a **single sample** we use the following notation:

- Input: `a0 ∈ R^2` (also written as `[x, y]`).
- Layer 1:
  - Weights: `W1 ∈ R^{Hidden1 × InputDim}` (4×2).
  - Biases: `b1 ∈ R^{Hidden1}`.
  - Pre-activation: `z1 = W1 · a0 + b1`.
  - Activation: `a1 = ReLU(z1)`.
- Layer 2:
  - Weights: `W2 ∈ R^{Hidden2 × Hidden1}` (8×4).
  - Biases: `b2 ∈ R^{Hidden2}`.
  - Pre-activation: `z2 = W2 · a1 + b2`.
  - Activation: `a2 = ReLU(z2)`.
- Output layer:
  - Weights: `W3 ∈ R^{OutputDim × Hidden2}` (2×8).
  - Biases: `b3 ∈ R^{OutputDim}`.
  - Logits: `z3 = W3 · a2 + b3`.
  - Probabilities: `p = softmax(z3)`.

In the code, all matrices are stored as **1D vectors**. The helper index function in `ToyNet.cpp` is:

```cpp
inline int idx(int row, int col, int cols) {
    return row * cols + col;
}
```

So, for example, `W1[idx(j, i, InputDim)]` accesses the weight from input unit `i` to hidden-1 unit `j`.

---

## 3. Forward pass (single sample)

See `ToyNet::forwardSingleWithActivations`.

### 3.1 Input layer

We start with an input point `(x, y)`:

```cpp
float a_in[InputDim] = {x, y};
```

This corresponds to the vector `a0`.

### 3.2 Layer 1 (4 units, ReLU)

For each hidden-1 unit `j`:

```cpp
float sum = b1[j];
for (int i = 0; i < InputDim; ++i) {
    sum += W1[j, i] * a_in[i];
}
a_h1[j] = relu(sum);
```

Mathematically:

```text
z1_j = Σ_i W1[j, i] * a0_i + b1_j
a1_j = ReLU(z1_j) = max(0, z1_j)
```

### 3.3 Layer 2 (8 units, ReLU)

For each hidden-2 unit `j`:

```text
z2_j = Σ_i W2[j, i] * a1_i + b2_j
a2_j = ReLU(z2_j)
```

Exactly the same pattern as layer 1, just with different dimensions.

### 3.4 Output layer (2 units, logits + softmax)

First compute logits:

```text
z3_k = Σ_j W3[k, j] * a2_j + b3_k
```

To get probabilities, we apply **softmax**. For numerical stability, we subtract the max logit before exponentiation:

```text
maxLogit = max_k z3_k

exp_k = exp(z3_k − maxLogit)
expSum = Σ_k exp_k
p_k = exp_k / expSum
```

In code, this appears as `logitsLocal`, `maxLogit`, `probsLocal`, and a divide by `expSum`.

The outputs `p0` and `p1` are just `p[0]` and `p[1]`. The predicted class is `argmax_k p_k`.

---

## 4. Loss and accuracy (batch)

The training function `ToyNet::trainBatch` handles a **mini-batch** of at most `MaxBatch` samples.

1. Inputs are copied into `m_a0` (shape: `batchSize × InputDim`).
2. The forward pass is run for all samples, filling
   - `m_z1`, `m_a1`, `m_z2`, `m_a2`, `m_logits`, `m_probs`.
3. For each sample, we compute **cross-entropy loss** and accuracy.

### 4.1 Cross-entropy loss with softmax

For a single sample with true label `y ∈ {0, 1}` and predicted probabilities `p`:

```text
L = −log(p_y)
```

To avoid `log(0)`, the code clamps `p_y` from below by a small epsilon:

```cpp
const float eps = 1e-6f;
lossSum += -std::log(std::max(correctProb, eps));
```

The batch loss is the average over the batch:

```text
L_batch = (1 / N) Σ_n L_n
```

where `N` is `batchSize`.

### 4.2 Accuracy

For each sample:

- Compute `predicted = argmax_k p_k`.
- Compare with the ground-truth label.
- Count how many are correct.

The batch accuracy is just `correct / N`.

`ToyNet::trainBatch` returns the batch loss and writes the batch accuracy to `outAccuracy`.

`Trainer` then stores these values in `lossHistory` and `accuracyHistory` for plotting.

---

## 5. Backpropagation (high level)

The code in `ToyNet::trainBatch` implements **manual backprop** rather than relying on an automatic differentiation library.

The key ideas:

1. For softmax + cross-entropy, the gradient of the loss w.r.t. the logits is:

   ```text
   δ3_k = dL/dz3_k = p_k − y_k
   ```

   where `y_k` is 1 for the correct class and 0 otherwise.

2. We propagate these gradients backwards through each linear+ReLU layer using the chain rule.

### 5.1 Output layer gradients

Given `δ3` for a single sample:

- Gradient for `W3`:

  ```text
  dL/dW3[k, j] += δ3_k * a2_j
  ```

- Gradient for `b3`:

  ```text
  dL/db3_k += δ3_k
  ```

- Gradient w.r.t. `a2` (to pass back to the previous layer):

  ```text
  δ2_raw_j = Σ_k δ3_k * W3[k, j]
  ```

In code, this is the loop that fills `m_dW3`, `m_db3`, and `delta2Raw`.

### 5.2 ReLU derivative at layer 2

Layer 2 uses ReLU:

```text
a2_j = ReLU(z2_j) = max(0, z2_j)
```

The derivative of ReLU is:

```text
ReLU'(z) = 1 if z > 0, 0 otherwise
```

So we apply:

```text
δ2_j = δ2_raw_j * 1(z2_j > 0)
```

This is exactly what the code does when constructing the `delta2` array using `m_z2`.

### 5.3 Layer 2 gradients

Now we treat layer 2 similarly to the output layer:

- Gradient for `W2`:

  ```text
  dL/dW2[j, i] += δ2_j * a1_i
  ```

- Gradient for `b2`:

  ```text
  dL/db2_j += δ2_j
  ```

- Gradient w.r.t. `a1`:

  ```text
  δ1_raw_i = Σ_j δ2_j * W2[j, i]
  ```

Again, the code accumulates these into `m_dW2`, `m_db2`, and `delta1Raw`.

### 5.4 ReLU derivative at layer 1 and its gradients

Apply ReLU derivative for layer 1:

```text
δ1_i = δ1_raw_i * 1(z1_i > 0)
```

Then compute gradients for `W1` and `b1`:

```text
dL/dW1[i, d] += δ1_i * a0_d

dL/db1_i += δ1_i
```

This fills `m_dW1` and `m_db1`.

### 5.5 Averaging gradients and SGD update

After processing all samples in the batch:

1. Gradients are averaged by multiplying by `1 / N`.
2. Parameters are updated with standard stochastic gradient descent:

```text
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
W3 -= lr * dW3
b3 -= lr * db3
```

This is exactly what the last loops in `ToyNet::trainBatch` implement.

`Trainer` controls the learning rate via `ToyNet::setLearningRate` and exposes a `learningRate` slider to the UI.

---

## 6. Batching and training control (`Trainer`)

`Trainer` adds a thin layer around `ToyNet`:

- Keeps a current batch (`m_batch`) and a cursor into the dataset.
- Uses `batchSize` to decide how many points to sample per step.
- Calls `ToyNet::trainBatch` with that batch.
- Stores `lastLoss` and `lastAccuracy`, plus history arrays for plotting in the ImGui UI.
- Supports:
  - Single-step training (`stepOnce`).
  - Auto-training with stopping conditions (`stepAuto`).

This is where **batch size** and **learning rate** become interactive knobs in the UI. Changing these parameters lets you see how they affect convergence and the decision boundary.

---

## 7. Relation to the GPU version

The math used here is **identical** to the one used in the fragment shader `shaders/field.frag`:

- Same architecture constants: `INPUT_DIM`, `HIDDEN1`, `HIDDEN2`, `OUTPUT_DIM`.
- Same indexing pattern for weights (`row * cols + col`).
- Same sequence:
  - Dense + ReLU → Dense + ReLU → Dense + softmax.

The big difference is that:

- On the **CPU**, `ToyNet::trainBatch` loops over a small batch of data points and computes gradients for training.
- On the **GPU**, the fragment shader loops over units of the network but is run **once per fragment** in parallel, purely for inference/visualization.

Understanding the CPU math first makes it much easier to understand how the GLSL version in the shader is doing the same thing, just massively parallelized across pixels.
