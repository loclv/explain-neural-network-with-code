/**
 * Neural Network Engine — TypeScript implementation matching the Zig code.
 *
 * This file is the mathematical core of the demo.  Every operation is
 * written out explicitly (triple-nested loops, no BLAS) so the steps are
 * easy to trace while you watch the visualisation.
 *
 * Architecture: fully-connected feedforward network.
 * Training:     stochastic gradient descent (SGD) with backpropagation.
 */

// ---------------------------------------------------------------------------
// MATRIX — row-major dense matrix stored as a flat Float32Array.
//
// Element (r, c) lives at index `r * cols + c`.  Every matrix function
// writes into a pre-allocated output buffer so there are no hidden
// allocations inside the hot training loop.
// ---------------------------------------------------------------------------

export type Matrix = {
	rows: number;
	cols: number;
	data: Float32Array;
};

/** Allocate a zero-initialised matrix with the given shape. */
export function mat(rows: number, cols: number): Matrix {
	return {
		rows,
		cols,
		data: new Float32Array(rows * cols),
	};
}

/** Convenience helper: build a matrix from a plain JS array. */
export function matFrom(rows: number, cols: number, values: number[]): Matrix {
	const m = mat(rows, cols);
	for (let i = 0; i < values.length; i++) m.data[i] = values[i];
	return m;
}

/** Row-major accessor: m[r, c]. */
export function matGet(m: Matrix, r: number, c: number): number {
	return m.data[r * m.cols + c];
}

/** Row-major setter: m[r, c] = v. */
export function matSet(m: Matrix, r: number, c: number, v: number): void {
	m.data[r * m.cols + c] = v;
}

/**
 * Matrix multiplication: out = a · b
 *
 * Classic triple loop.  a must be (m × n), b must be (n × p),
 * and out must be pre-allocated as (m × p).
 */
export function matDot(out: Matrix, a: Matrix, b: Matrix): void {
	for (let i = 0; i < a.rows; i++) {
		for (let j = 0; j < b.cols; j++) {
			let sum = 0;
			// Inner dot-product over the shared dimension.
			for (let k = 0; k < a.cols; k++) {
				sum += matGet(a, i, k) * matGet(b, k, j);
			}
			matSet(out, i, j, sum);
		}
	}
}

/** Element-wise addition: out = a + b.  All matrices must have identical shape. */
export function matAdd(out: Matrix, a: Matrix, b: Matrix): void {
	for (let i = 0; i < a.data.length; i++) {
		out.data[i] = a.data[i] + b.data[i];
	}
}

/** Copy contents of a into out. */
export function matCopy(out: Matrix, a: Matrix): void {
	out.data.set(a.data);
}

/** Apply a scalar function to every element: out[i] = fn(a[i]). */
export function matApply(
	out: Matrix,
	a: Matrix,
	fn: (x: number) => number,
): void {
	for (let i = 0; i < a.data.length; i++) {
		out.data[i] = fn(a.data[i]);
	}
}

// ---------------------------------------------------------------------------
// ACTIVATION FUNCTIONS
//
// We use Leaky ReLU for hidden layers (avoids "dead neurons") and sigmoid
// for the output layer (gives us a probability-like value for binary
// classification).
// ---------------------------------------------------------------------------

/**
 * Leaky ReLU.
 *
 *   f(x) =  x        if x > 0
 *   f(x) =  0.01·x   otherwise
 *
 * The small negative slope (0.01) keeps gradients flowing even when the
 * neuron receives negative input, preventing the "dying ReLU" problem that
 * is common in tiny networks with unlucky initialisation.
 */
export function relu(x: number): number {
	return x > 0 ? x : 0.01 * x;
}

/** Derivative of Leaky ReLU needed for backpropagation. */
export function reluDeriv(x: number): number {
	return x > 0 ? 1 : 0.01;
}

/**
 * Sigmoid.
 *
 *   f(x) = 1 / (1 + e^(-x))
 *
 * Squashes any real number into the range (0, 1).  Perfect for binary
 * classification because the output can be interpreted as a probability.
 */
export function sigmoid(x: number): number {
	return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Derivative of sigmoid.
 *
 *   f'(x) = f(x) · (1 - f(x))
 *
 * This elegant identity avoids computing exp() again in the backward pass.
 */
export function sigmoidDeriv(x: number): number {
	const s = sigmoid(x);
	return s * (1.0 - s);
}

// ---------------------------------------------------------------------------
// LAYER — one fully-connected layer.
//
// A layer stores:
//   weights        — (outputSize × inputSize)  linear transform W
//   biases         — (outputSize × 1)          additive term b
//   preActivation  — cached z = W·a + b        needed for derivative
//   activation     — cached a = f(z)           passed to next layer
//   *_gradients    — reused buffers for dW, db (no alloc in training loop)
// ---------------------------------------------------------------------------

export interface Layer {
	weights: Matrix;
	biases: Matrix;
	preActivation: Matrix;
	activation: Matrix;
	weightGradients: Matrix;
	biasGradients: Matrix;
}

/** Allocate a layer with the given input/output dimensions. */
export function createLayer(inputSize: number, outputSize: number): Layer {
	return {
		weights: mat(outputSize, inputSize),
		biases: mat(outputSize, 1),
		preActivation: mat(outputSize, 1),
		activation: mat(outputSize, 1),
		weightGradients: mat(outputSize, inputSize),
		biasGradients: mat(outputSize, 1),
	};
}

/**
 * He initialisation.
 *
 * Weights are drawn from a uniform distribution scaled by sqrt(2 / fan_in).
 * This variance-preserving rule keeps the magnitude of activations roughly
 * constant across layers, making training much more stable than naive
 * random initialisation.
 *
 * We use a tiny Linear-Congruential-Generator (LCG) so the demo is fully
 * deterministic — every page reload starts from the exact same weights.
 */
export function randomizeLayer(layer: Layer, seed = 42): void {
	let s = seed;
	const next = () => {
		s = (s * 1103515245 + 12345) & 0x7fffffff;
		return s / 0x7fffffff;
	};
	const scale = Math.sqrt(2.0 / layer.weights.cols); // He scale
	for (let i = 0; i < layer.weights.data.length; i++) {
		layer.weights.data[i] = (next() - 0.5) * 2.0 * scale;
	}
	for (let i = 0; i < layer.biases.data.length; i++) {
		layer.biases.data[i] = (next() - 0.5) * 2.0 * scale;
	}
}

/**
 * Forward pass for a single layer.
 *
 *   z = W · input + b
 *   a = activation(z)
 *
 * Both z (preActivation) and a (activation) are cached because we need
 * them again during backpropagation.
 */
export function forwardLayer(
	layer: Layer,
	input: Matrix,
	actFn: (x: number) => number,
): void {
	matDot(layer.preActivation, layer.weights, input); // z = W·x
	matAdd(layer.preActivation, layer.preActivation, layer.biases); // z += b
	matApply(layer.activation, layer.preActivation, actFn); // a = f(z)
}

// ---------------------------------------------------------------------------
// NEURAL NETWORK — a stack of layers.
// ---------------------------------------------------------------------------

export interface NeuralNetwork {
	layers: Layer[];
}

/** Build a network from a list of layer sizes, e.g. [2, 4, 1]. */
export function createNetwork(layerSizes: number[]): NeuralNetwork {
	const layers: Layer[] = [];
	for (let i = 0; i < layerSizes.length - 1; i++) {
		layers.push(createLayer(layerSizes[i], layerSizes[i + 1]));
	}
	return { layers };
}

/** Randomise every layer with consecutive seeds. */
export function randomizeNetwork(network: NeuralNetwork, seed = 42): void {
	for (const layer of network.layers) {
		randomizeLayer(layer, seed++);
	}
}

/**
 * Predict — pure forward inference.
 *
 * Data flows left-to-right through every layer.
 * Hidden layers use Leaky ReLU; the final layer always uses sigmoid.
 */
export function predict(network: NeuralNetwork, input: Matrix): Matrix {
	let current = input;
	for (let i = 0; i < network.layers.length; i++) {
		const actFn = i === network.layers.length - 1 ? sigmoid : relu;
		forwardLayer(network.layers[i], current, actFn);
		current = network.layers[i].activation;
	}
	return current; // the final layer's activation matrix
}

// ---------------------------------------------------------------------------
// BACKPROPAGATION — the core learning algorithm.
// ---------------------------------------------------------------------------

/**
 * Perform one training step (one forward + one backward pass) on a single
 * (input, target) pair and return the mean-squared-error loss.
 *
 * The logic is split into four phases:
 *
 *   1. FORWARD   — compute predictions, cache z and a for every layer.
 *   2. LOSS      — MSE = average of (output - target)².
 *   3. BACKWARD  — propagate error gradients from output back to input.
 *   4. UPDATE    — move weights in the direction that reduces loss.
 */
export function trainStep(
	network: NeuralNetwork,
	input: Matrix,
	target: Matrix,
	learningRate: number,
): number {
	// ========== 1. FORWARD PASS ==========
	let current = input;
	for (let i = 0; i < network.layers.length; i++) {
		const actFn = i === network.layers.length - 1 ? sigmoid : relu;
		forwardLayer(network.layers[i], current, actFn);
		current = network.layers[i].activation;
	}

	const output = network.layers[network.layers.length - 1].activation;

	// ========== 2. LOSS ==========
	// Mean Squared Error: how far are our predictions from the truth?
	let loss = 0;
	for (let i = 0; i < target.data.length; i++) {
		const diff = output.data[i] - target.data[i];
		loss += diff * diff; // square so positive and negative errors both count
	}
	loss /= target.data.length;

	// ========== 3. BACKWARD PASS ==========
	//
	// We compute ∂Loss/∂W for every weight using the chain rule.
	//
	//   output error = (a_L - y) ⊙ f'(z_L)
	//
	//   hidden error = (W_{l+1}ᵀ · δ_{l+1}) ⊙ f'(z_l)
	//
	// ⊙ = element-wise multiplication.
	//
	// The key insight: each layer only needs the "error signal" δ from the
	// layer to its right, so we can compute everything in one reverse sweep.

	// ---- Output layer error ----
	const outLayer = network.layers[network.layers.length - 1];
	matCopy(outLayer.biasGradients, output);
	for (let i = 0; i < outLayer.biasGradients.data.length; i++) {
		outLayer.biasGradients.data[i] -= target.data[i]; // δ = a - y
	}
	// Multiply by the derivative of the activation (chain rule).
	for (let i = 0; i < outLayer.biasGradients.data.length; i++) {
		outLayer.biasGradients.data[i] *= sigmoidDeriv(
			outLayer.preActivation.data[i],
		);
	}

	// ---- Hidden layers (propagate backwards) ----
	if (network.layers.length > 1) {
		let nextError = outLayer.biasGradients;
		let nextWeights = outLayer.weights;

		for (let li = network.layers.length - 1; li > 0; li--) {
			const layer = network.layers[li - 1];

			// Transpose the weights of the layer to the right so we can
			// "send the error signal backwards" through the connections.
			const wT = mat(nextWeights.cols, nextWeights.rows);
			for (let i = 0; i < nextWeights.rows; i++) {
				for (let j = 0; j < nextWeights.cols; j++) {
					matSet(wT, j, i, matGet(nextWeights, i, j));
				}
			}

			// δ_l = W_{l+1}ᵀ · δ_{l+1}
			const errorBeforeAct = mat(wT.rows, nextError.cols);
			matDot(errorBeforeAct, wT, nextError);

			// δ_l = δ_l ⊙ f'(z_l)
			matCopy(layer.biasGradients, errorBeforeAct);
			for (let i = 0; i < layer.biasGradients.data.length; i++) {
				layer.biasGradients.data[i] *= reluDeriv(layer.preActivation.data[i]);
			}

			// Move one step left for the next iteration.
			nextError = layer.biasGradients;
			nextWeights = layer.weights;
		}
	}

	// ========== 4. PARAMETER UPDATE ==========
	//
	// Gradient descent: W_new = W_old - lr · ∂Loss/∂W
	//
	// For a dense layer ∂Loss/∂W = δ · a_prevᵀ, so we need the transpose of
	// the previous layer's activation.

	let prevAct = input;
	for (const layer of network.layers) {
		// pT = prevActᵀ  (so we can do outer product: dW = δ · aᵀ)
		const pT = mat(prevAct.cols, prevAct.rows);
		for (let i = 0; i < prevAct.rows; i++) {
			for (let j = 0; j < prevAct.cols; j++) {
				matSet(pT, j, i, matGet(prevAct, i, j));
			}
		}
		matDot(layer.weightGradients, layer.biasGradients, pT);

		// W ← W - lr · dW
		for (let i = 0; i < layer.weights.data.length; i++) {
			layer.weights.data[i] -= learningRate * layer.weightGradients.data[i];
		}
		// b ← b - lr · db
		for (let i = 0; i < layer.biases.data.length; i++) {
			layer.biases.data[i] -= learningRate * layer.biasGradients.data[i];
		}

		prevAct = layer.activation; // shift right for the next layer
	}

	return loss;
}

/**
 * Train the network for multiple epochs using Stochastic Gradient Descent.
 *
 * Every epoch we loop over the full dataset (the four XOR examples) and
 * perform one trainStep per example.  We report the average loss every
 * 100 epochs via the onEpoch callback so the UI can draw the loss curve.
 */
export function train(
	network: NeuralNetwork,
	inputs: Matrix[],
	targets: Matrix[],
	epochs: number,
	learningRate: number,
	onEpoch?: (epoch: number, avgLoss: number) => void,
): void {
	for (let epoch = 0; epoch < epochs; epoch++) {
		let totalLoss = 0;
		for (let i = 0; i < inputs.length; i++) {
			totalLoss += trainStep(network, inputs[i], targets[i], learningRate);
		}
		const avgLoss = totalLoss / inputs.length;
		if (onEpoch && epoch % 100 === 0) {
			onEpoch(epoch, avgLoss);
		}
	}
}

// ---------------------------------------------------------------------------
// XOR DATASET — the classic non-linearly-separable problem.
//
// A single-layer perceptron cannot solve XOR because the classes are not
// linearly separable.  Our 2-4-1 network learns two linear boundaries
// in the hidden layer and combines them in the output layer.
// ---------------------------------------------------------------------------

export const xorInputs = [
	matFrom(2, 1, [0, 0]),
	matFrom(2, 1, [0, 1]),
	matFrom(2, 1, [1, 0]),
	matFrom(2, 1, [1, 1]),
];

export const xorTargets = [
	matFrom(1, 1, [0]),
	matFrom(1, 1, [1]),
	matFrom(1, 1, [1]),
	matFrom(1, 1, [0]),
];

// ---------------------------------------------------------------------------
// GRID DATASET — 10×10 grid with 1 or 2 random points.
//
// The network must learn to classify: 1 point → class 0, 2 points → class 1.
// Each input is a flat 100-element vector (row-major).
// ---------------------------------------------------------------------------

export const GRID_SIZE = 10;
export const GRID_INPUT = GRID_SIZE * GRID_SIZE;

/** Build a flat 100×1 input matrix with `points` random cells set to 1. */
export function makeGridInput(rng: () => number, points: number): Matrix {
	const m = mat(GRID_INPUT, 1);
	let placed = 0;
	while (placed < points) {
		const idx = Math.floor(rng() * GRID_INPUT);
		if (m.data[idx] === 0) {
			m.data[idx] = 1.0;
			placed++;
		}
	}
	return m;
}

/** Generate N random training samples (half 1-point, half 2-point). */
export function generateGridSamples(
	rng: () => number,
	n: number,
): { inputs: Matrix[]; targets: Matrix[] } {
	const inputs: Matrix[] = [];
	const targets: Matrix[] = [];
	for (let i = 0; i < n; i++) {
		const pts = i % 2 === 0 ? 1 : 2;
		inputs.push(makeGridInput(rng, pts));
		targets.push(matFrom(1, 1, [pts === 1 ? 0 : 1]));
	}
	return { inputs, targets };
}

/** Simple LCG for deterministic grid generation in the browser. */
export function gridRng(seed = 42) {
	let s = seed;
	return () => {
		s = (s * 1103515245 + 12345) & 0x7fffffff;
		return s / 0x7fffffff;
	};
}
