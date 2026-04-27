# Architecture

## Overview

This is a single-file library (`src/root.zig`) implementing a fully-connected feedforward neural network from scratch. It deliberately avoids abstraction layers to keep the code traceable for educational purposes.

## Core Types

### Matrix

Row-major dense matrix. All fields are public for transparency, but `data` is a flat `[]f32` slice indexed by `r * cols + c`.

All matrix operations are **out-place** (write to a pre-allocated output buffer). This simplifies memory management at the cost of some verbosity at call sites.

### Layer

A dense layer containing:
- `weights` (output_size x input_size)
- `biases` (output_size x 1)
- `pre_activation` (output_size x 1) — cached z = Wx + b
- `activation` (output_size x 1) — cached a = f(z)
- `weight_gradients` — reused buffer for dW
- `bias_gradients` — reused buffer for db

All buffers are allocated once at layer construction and reused across every training step.

### NeuralNetwork

A slice of `Layer` structs, plus a reference to the allocator used to build them. The network is a simple stack: input flows through layer 0, then layer 1, etc.

The last layer always uses sigmoid; every hidden layer uses Leaky ReLU.

## Data Flow

```
Input (n x 1)
  -> Layer 0: dot(W0, input) + b0 -> Leaky ReLU -> activation0
  -> Layer 1: dot(W1, activation0) + b1 -> sigmoid -> output
```

## Training Loop (SGD)

For each epoch:
1. For each (input, target) pair:
   - **Forward pass**: compute every layer's `pre_activation` and `activation`
   - **Backward pass**:
     - Output error: `(output - target) * sigmoid'(pre_activation)`
     - Propagate backward through hidden layers: `W^T * error * ReLU'(pre_activation)`
   - **Parameter update**: `W -= lr * gradient`, `b -= lr * gradient`

All gradients overwrite the pre-allocated buffers inside each `Layer`.

## Memory Model

- Caller provides an `Allocator` to `NeuralNetwork.init()`
- `NeuralNetwork.deinit()` frees all layer buffers and the layer slice itself
- `Matrix.deinit()` frees only the `data` slice
- No hidden allocations during `predict()` or inside the inner training loop

## Why These Choices

- **No generics**: Everything uses `f32`. Keeps the code readable and avoids monomorphization bloat.
- **No BLAS / SIMD**: The matrix multiply is a triple-nested loop. Fine for small toy problems; easy to swap later.
- **Leaky ReLU instead of ReLU**: Prevents "dead neurons" during XOR training, which commonly occurs with small networks and unlucky initialization.
- **He initialization**: `scale = sqrt(2 / fan_in)` for ReLU layers.
