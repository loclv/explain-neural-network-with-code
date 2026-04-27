/**
 * NetworkDiagram — SVG visualisation of a fully-connected feedforward network.
 *
 * Renders neurons as circles and weights as coloured lines.
 *   Green  = positive weight
 *   Red    = negative weight
 *   Thickness & opacity ∝ |weight|
 *   Neuron fill intensity ∝ activation magnitude
 *
 * The layout is hard-coded for a 2-4-1 architecture because that is the
 * XOR demo network.  Generalising to arbitrary topologies is left as an
 * exercise for the reader. 🙂
 */

import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { Layer } from '../nn-engine';

interface Props {
  layers: Layer[];
  inputValues: number[];
  highlightActive?: boolean;
}

interface NeuronPos {
  x: number;
  y: number;
  layer: number;
  idx: number;
  value: number;
  label: string;
}

export default function NetworkDiagram({
  layers,
  inputValues,
  highlightActive = true,
}: Props) {
  const { t } = useTranslation();
  const neuronPositions = useMemo(() => {
    const positions: NeuronPos[] = [];
    const layerCounts = [2, ...layers.map((l) => l.weights.rows)];
    const layerX = [80, 280, 480];

    for (let li = 0; li < layerCounts.length; li++) {
      const count = layerCounts[li];
      const spacing = 120 / (count + 1);
      for (let ni = 0; ni < count; ni++) {
        const y = 40 + (ni + 1) * spacing;
        let value = 0;
        let label = '';
        if (li === 0) {
          value = inputValues[ni] ?? 0;
          label = `x${ni + 1}`;
        } else {
          value = layers[li - 1].activation.data[ni] ?? 0;
          label = li === layerCounts.length - 1 ? 'y' : `h${ni + 1}`;
        }
        positions.push({
          x: layerX[li],
          y,
          layer: li,
          idx: ni,
          value,
          label,
        });
      }
    }
    return positions;
  }, [layers, inputValues]);

  const connections = useMemo(() => {
    const conns: {
      from: NeuronPos;
      to: NeuronPos;
      weight: number;
    }[] = [];

    for (let li = 0; li < layers.length; li++) {
      const fromLayer = neuronPositions.filter((n) => n.layer === li);
      const toLayer = neuronPositions.filter((n) => n.layer === li + 1);
      const layer = layers[li];

      for (const to of toLayer) {
        for (const from of fromLayer) {
          conns.push({
            from,
            to,
            weight: layer.weights.data[to.idx * layer.weights.cols + from.idx],
          });
        }
      }
    }
    return conns;
  }, [layers, neuronPositions]);

  const getNeuronColor = (value: number, isOutput: boolean) => {
    if (!highlightActive) return '#334155';
    const intensity = Math.abs(value);
    const r = isOutput
      ? Math.round(255 * intensity)
      : Math.round(128 + 127 * (value > 0 ? intensity : -intensity));
    const g = isOutput
      ? Math.round(128 - 64 * intensity)
      : Math.round(128 + 127 * (value > 0 ? intensity : 0));
    const b = isOutput
      ? Math.round(128 - 64 * intensity)
      : Math.round(128 + 127 * (value < 0 ? intensity : 0));
    return `rgb(${Math.max(0, Math.min(255, r))}, ${Math.max(0, Math.min(255, g))}, ${Math.max(0, Math.min(255, b))})`;
  };

  return (
    <svg
      viewBox="0 0 560 200"
      className="w-full h-auto border border-slate-200 rounded-lg bg-slate-50"
    >
      <title>Neural network architecture diagram</title>
      {/* Connections */}
      {connections.map((conn) => {
        const isPositive = conn.weight >= 0;
        const thickness = Math.max(0.5, Math.min(3, Math.abs(conn.weight) * 5));
        const opacity = Math.max(0.1, Math.min(1, Math.abs(conn.weight) * 2));
        return (
          <line
            key={`conn-${conn.from.layer}-${conn.from.idx}-${conn.to.layer}-${conn.to.idx}`}
            x1={conn.from.x}
            y1={conn.from.y}
            x2={conn.to.x}
            y2={conn.to.y}
            stroke={isPositive ? '#22c55e' : '#ef4444'}
            strokeWidth={thickness}
            opacity={opacity}
          />
        );
      })}

      {/* Neurons */}
      {neuronPositions.map((n) => (
        <g key={`neuron-${n.layer}-${n.idx}`}>
          <circle
            cx={n.x}
            cy={n.y}
            r={18}
            fill={getNeuronColor(
              n.value,
              n.layer === neuronPositions[neuronPositions.length - 1].layer,
            )}
            stroke="#475569"
            strokeWidth={2}
          />
          <text
            x={n.x}
            y={n.y + 5}
            textAnchor="middle"
            fontSize={12}
            fill="white"
            fontWeight="bold"
          >
            {n.label}
          </text>
          <text
            x={n.x}
            y={n.y + 32}
            textAnchor="middle"
            fontSize={10}
            fill="#64748b"
          >
            {n.value.toFixed(3)}
          </text>
        </g>
      ))}

      {/* Layer labels */}
      <text x={80} y={190} textAnchor="middle" fontSize={11} fill="#64748b">
        {t('network.input')}
      </text>
      <text x={280} y={190} textAnchor="middle" fontSize={11} fill="#64748b">
        {t('network.hidden')}
      </text>
      <text x={480} y={190} textAnchor="middle" fontSize={11} fill="#64748b">
        {t('network.output')}
      </text>
    </svg>
  );
}
