# Neural Network from Scratch in Zig

A minimal, fully-connected feedforward neural network implemented from scratch in Zig. No external dependencies. No GPU code. Just matrices, backpropagation, and a working XOR solver.

## Features

- **Dense Matrix type** with row-major storage and basic linear algebra (`dot`, `transpose`, `add`, element-wise ops)
- **Feedforward layers** with configurable input/output sizes
- **Activation functions**: Leaky ReLU (hidden) and sigmoid (output), with derivatives
- **Full backpropagation** training loop via stochastic gradient descent
- **He initialization** for weights
- **Built-in tests** including an end-to-end XOR convergence test
- **Zero dependencies** beyond the Zig standard library

## Requirements

- Zig `0.16.0` or compatible

## Quick Start

```bash
# Run the XOR demo
zig build run

# Run all tests
zig build test
```

## Project Structure

```
src/
  root.zig      # Library: Matrix, Layer, NeuralNetwork, activation functions
  main.zig      # CLI demo: trains 2-4-1 network on XOR
build.zig       # Build script
build.zig.zon   # Package manifest
```

## API Overview

### Matrix

```zig
var m = try Matrix.init(allocator, rows, cols);
defer m.deinit(allocator);
m.set(r, c, value);
const v = m.get(r, c);
```

Static methods for linear algebra (all operate on pre-allocated output buffers):

```zig
Matrix.dot(&out, a, b);
Matrix.add(&out, a, b);
Matrix.transpose(&out, a);
Matrix.elementWiseMul(&out, a, b);
Matrix.copy(&out, a);
Matrix.apply(&out, a, &my_activation);
```

### NeuralNetwork

```zig
const layer_sizes = [_]usize{ 2, 4, 1 };
var nn = try NeuralNetwork.init(allocator, &layer_sizes);
defer nn.deinit();

nn.randomize(rng);

const output = nn.predict(input);
try nn.train(&inputs, &targets, epochs, learning_rate);
```

### Activation Functions

```zig
pub fn relu(x: f32) f32;
pub fn relu_derivative(x: f32) f32;
pub fn sigmoid(x: f32) f32;
pub fn sigmoid_derivative(x: f32) f32;
```

## Design Notes

- **Manual memory management**: Every `Matrix` and `NeuralNetwork` requires an allocator and a matching `deinit()` call.
- **Pre-allocated buffers**: The training loop reuses layer-internal gradient buffers to minimize allocations.
- **Fixed architecture**: Network topology is set at init time. The last layer always uses sigmoid; all hidden layers use Leaky ReLU.
- **Single-sample SGD**: `train()` updates weights after every example. No batching, no momentum, no Adam.

## Limitations / Future Work

- No mini-batch or full-batch gradient descent
- No optimizer extensions (momentum, Adam, RMSprop)
- No softmax / cross-entropy (binary sigmoid only)
- No serialization (save/load weights)
- No dropout, batch norm, or regularization
- Single-threaded CPU only

## License

Public domain / Unlicense (or specify your own).
