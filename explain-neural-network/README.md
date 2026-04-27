# Neural Network Visualiser

An interactive web application that explains how a feedforward neural network learns the XOR problem. Built with React, TypeScript, Tailwind CSS, and Rsbuild.

## What it shows

This demo implements the exact same algorithm as the Zig neural network in the parent repository, but renders it in the browser so you can watch the learning process in real time.

### Features

- **Live network diagram** — neurons and connections update as you drag the input sliders or train the network
- **Interactive inputs** — two sliders (0–1) feed values into the network and show the prediction instantly
- **XOR truth table** — four cards that turn green as the network learns each case
- **Training animation** — click "Train on XOR" and watch 10,000 epochs of stochastic gradient descent unfold in 100-epoch chunks
- **Loss curve** — a canvas-drawn graph of mean-squared error over time
- **Weight matrices** — live tables showing every weight value, colour-coded positive (green) and negative (red)
- **Code explorer** — annotated source-code panels that walk through matrix multiplication, activation functions, He initialisation, forward pass, backpropagation, and gradient descent

## The algorithm

- **Architecture**: 2-4-1 fully-connected feedforward network
- **Hidden activation**: Leaky ReLU (slope 0.01 for negative inputs)
- **Output activation**: Sigmoid (probability-like output in (0, 1))
- **Initialisation**: He uniform (scaled by sqrt(2 / fan_in))
- **Loss**: Mean squared error
- **Optimiser**: Stochastic gradient descent, learning rate 0.1
- **Training**: 10,000 epochs on the four XOR examples, updating weights after every example

## Why XOR?

XOR is the classic example of a problem that is *not* linearly separable. A single-layer perceptron cannot solve it. By adding one hidden layer with four neurons, the network learns to draw two linear boundaries in the input space and combine them in the output layer — demonstrating the power of depth in neural networks.

## Running locally

```bash
bun install
bun run dev        # http://localhost:3000
bun run build      # production bundle in dist/
```

## File guide

| File | Purpose |
|------|---------|
| `src/nn-engine.ts` | Matrix ops, activations, forward & backward passes, SGD loop |
| `src/components/NetworkDiagram.tsx` | SVG rendering of neurons and weighted connections |
| `src/components/CodeExplorer.tsx` | Accordion panels showing annotated engine code |
| `src/App.tsx` | Main UI: sliders, truth table, loss chart, weight inspector |
| `src/App.css` | Global styles (mostly Tailwind) |
