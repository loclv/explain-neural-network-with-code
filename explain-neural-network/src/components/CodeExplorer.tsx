import { useState } from 'react';
import { useTranslation } from 'react-i18next';

interface CodeBlock {
  title: string;
  description: string;
  code: string;
}

function getBlocks(t: (key: string) => string): CodeBlock[] {
  return [
    {
      title: t('code.blocks.matrixDefinition.title'),
      description: t('code.blocks.matrixDefinition.description'),
      code: `export type Matrix = {
  rows: number;
  cols: number;
  data: Float32Array;
};

export function mat(rows: number, cols: number): Matrix {
  return {
    rows,
    cols,
    data: new Float32Array(rows * cols),
  };
}`,
    },
    {
      title: t('code.blocks.matrixMultiplication.title'),
      description: t('code.blocks.matrixMultiplication.description'),
      code: `export function matDot(out: Matrix, a: Matrix, b: Matrix): void {
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += matGet(a, i, k) * matGet(b, k, j);
      }
      matSet(out, i, j, sum);
    }
  }
}`,
    },
    {
      title: t('code.blocks.activationFunctions.title'),
      description: t('code.blocks.activationFunctions.description'),
      code: `/**
 * Leaky ReLU: f(x) = x if x > 0, else 0.01·x
 * The small negative slope keeps gradients flowing.
 */
export function relu(x: number): number {
  return x > 0 ? x : 0.01 * x;
}

/**
 * Sigmoid: f(x) = 1 / (1 + e^(-x))
 * Squashes any value into (0, 1).
 */
export function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}`,
    },
    {
      title: t('code.blocks.heInitialisation.title'),
      description: t('code.blocks.heInitialisation.description'),
      code: `export function randomizeLayer(layer: Layer, seed = 42): void {
  let s = seed;
  const next = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  // He scale: sqrt(2 / input_size)
  const scale = Math.sqrt(2.0 / layer.weights.cols);

  for (let i = 0; i < layer.weights.data.length; i++) {
    layer.weights.data[i] = (next() - 0.5) * 2.0 * scale;
  }
}`,
    },
    {
      title: t('code.blocks.forwardPass.title'),
      description: t('code.blocks.forwardPass.description'),
      code: `export function forwardLayer(
  layer: Layer,
  input: Matrix,
  actFn: (x: number) => number,
): void {
  // z = W · x
  matDot(layer.preActivation, layer.weights, input);
  // z = z + b
  matAdd(layer.preActivation, layer.preActivation, layer.biases);
  // a = f(z)
  matApply(layer.activation, layer.preActivation, actFn);
}`,
    },
    {
      title: t('code.blocks.backpropOutput.title'),
      description: t('code.blocks.backpropOutput.description'),
      code: `// Output error = (prediction - target) * f'(z)
const outLayer = network.layers[network.layers.length - 1];
matCopy(outLayer.biasGradients, output);

// δ = a - y  (difference from target)
for (let i = 0; i < outLayer.biasGradients.data.length; i++) {
  outLayer.biasGradients.data[i] -= target.data[i];
}

// δ = δ ⊙ f'(z)  (chain rule)
for (let i = 0; i < outLayer.biasGradients.data.length; i++) {
  outLayer.biasGradients.data[i] *= sigmoidDeriv(
    outLayer.preActivation.data[i],
  );
}`,
    },
    {
      title: t('code.blocks.backpropHidden.title'),
      description: t('code.blocks.backpropHidden.description'),
      code: `// Transpose W so we can send error backwards
const wT = mat(nextWeights.cols, nextWeights.rows);
for (let i = 0; i < nextWeights.rows; i++) {
  for (let j = 0; j < nextWeights.cols; j++) {
    matSet(wT, j, i, matGet(nextWeights, i, j));
  }
}

// δ_l = W_{l+1}ᵀ · δ_{l+1}
matDot(errorBeforeAct, wT, nextError);

// δ_l = δ_l ⊙ f'(z_l)
matCopy(layer.biasGradients, errorBeforeAct);
for (let i = 0; i < layer.biasGradients.data.length; i++) {
  layer.biasGradients.data[i] *= reluDeriv(
    layer.preActivation.data[i],
  );
}`,
    },
    {
      title: t('code.blocks.gradientDescent.title'),
      description: t('code.blocks.gradientDescent.description'),
      code: `// dW = δ · a_prevᵀ
matDot(layer.weightGradients, layer.biasGradients, pT);

// W ← W - lr · dW
for (let i = 0; i < layer.weights.data.length; i++) {
  layer.weights.data[i] -= learningRate * layer.weightGradients.data[i];
}

// b ← b - lr · db
for (let i = 0; i < layer.biases.data.length; i++) {
  layer.biases.data[i] -= learningRate * layer.biasGradients.data[i];
}`,
    },
  ];
}

export default function CodeExplorer() {
  const { t } = useTranslation();
  const [open, setOpen] = useState<number | null>(0);
  const blocks = getBlocks(t);

  return (
    <div className="space-y-4">
      {blocks.map((block, i) => {
        const isOpen = open === i;
        return (
          <div
            key={block.title}
            className="border border-slate-200 rounded-lg overflow-hidden"
          >
            <button
              type="button"
              onClick={() => setOpen(isOpen ? null : i)}
              className="w-full flex items-center justify-between p-4 bg-white hover:bg-slate-50 transition text-left"
            >
              <div>
                <h3 className="font-semibold text-slate-800">{block.title}</h3>
                <p className="text-sm text-slate-500 mt-1 line-clamp-2">
                  {block.description}
                </p>
              </div>
              <svg
                aria-hidden="true"
                className={`w-5 h-5 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <title>{t('code.expandSection')}</title>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
            {isOpen && (
              <div className="bg-slate-900 text-slate-50 p-4 overflow-x-auto">
                <pre className="text-sm font-mono leading-relaxed">
                  <code>{block.code}</code>
                </pre>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
