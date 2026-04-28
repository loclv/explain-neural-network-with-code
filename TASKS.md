# Tasks

## Completed

- [x] Implement `Matrix` type with row-major storage
- [x] Add matrix operations: `dot`, `add`, `transpose`, `scale`, `elementWiseMul`, `copy`, `apply`
- [x] Implement activation functions: Leaky ReLU and sigmoid with derivatives
- [x] Implement `Layer` struct (weights, biases, pre-activation, activation, gradients)
- [x] Implement `NeuralNetwork` struct with init/deinit
- [x] Implement forward pass with configurable per-layer activation
- [x] Implement full backpropagation training loop (SGD)
- [x] Add He weight initialization via randomization
- [x] Add matrix math unit test
- [x] Add end-to-end XOR convergence test
- [x] Add CLI demo in `main.zig` (XOR)
- [x] Replace CLI demo with 10×10 grid point-counter (1 point vs 2 points)
- [x] Fix Zig 0.16.0 compatibility (function pointer types, `std.Random` passing)
- [x] Fix prediction evaluation order (predictions were overwritten by subsequent calls)
- [x] Create README.md, TASKS.md, ARCHITECTURE.md

## Backlog / Future Work

- [ ] Mini-batch and full-batch gradient descent
- [ ] Optimizers: momentum, RMSprop, Adam
- [ ] Softmax activation + cross-entropy loss for multi-class
- [ ] Weight serialization (save/load from file)
- [ ] L1/L2 regularization
- [ ] Dropout (training-time random masking)
- [ ] Batch normalization
- [ ] Conv2d layer (CNN)
- [ ] Multi-threaded matrix multiplication
- [ ] Benchmarks vs a reference implementation
