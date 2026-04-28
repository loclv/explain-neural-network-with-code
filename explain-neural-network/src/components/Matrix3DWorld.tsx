import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface V3 {
  x: number;
  y: number;
  z: number;
}
function mv(m: number[][], v: V3): V3 {
  return {
    x: m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
    y: m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
    z: m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
  };
}
const CUBE: V3[] = [
  { x: -1, y: -1, z: -1 },
  { x: 1, y: -1, z: -1 },
  { x: 1, y: 1, z: -1 },
  { x: -1, y: 1, z: -1 },
  { x: -1, y: -1, z: 1 },
  { x: 1, y: -1, z: 1 },
  { x: 1, y: 1, z: 1 },
  { x: -1, y: 1, z: 1 },
];
const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];
const PS = [
  {
    k: 'I',
    l: 'Identity',
    m: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  },
  {
    k: 'SU',
    l: 'Scale ×1.5',
    m: [
      [1.5, 0, 0],
      [0, 1.5, 0],
      [0, 0, 1.5],
    ],
  },
  {
    k: 'SD',
    l: 'Scale ×0.5',
    m: [
      [0.5, 0, 0],
      [0, 0.5, 0],
      [0, 0, 0.5],
    ],
  },
  {
    k: 'RX',
    l: 'Rotate X',
    m: [
      [1, 0, 0],
      [0, 0, -1],
      [0, 1, 0],
    ],
  },
  {
    k: 'RY',
    l: 'Rotate Y',
    m: [
      [0, 0, 1],
      [0, 1, 0],
      [-1, 0, 0],
    ],
  },
  {
    k: 'RZ',
    l: 'Rotate Z',
    m: [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ],
  },
  {
    k: 'SH',
    l: 'Shear',
    m: [
      [1, 0.5, 0],
      [0.3, 1, 0],
      [0, 0, 1],
    ],
  },
];
const OX = 140,
  OY = 140,
  SCL = 60,
  FL = 300;
function proj(v: V3) {
  const d = FL + v.z * 40,
    s = FL / (d || 1);
  return { x: OX + v.x * SCL * s, y: OY - v.y * SCL * s };
}
export default function Matrix3DWorld() {
  const { t } = useTranslation();
  const [mat, setMat] = useState(PS[0].m.map((r) => [...r]));
  const [preset, setPreset] = useState('I');
  const [rx, setRx] = useState(25);
  const [sel, setSel] = useState(0);
  const drag = useRef({ on: false, lx: 0, ly: 0 });
  const tf = CUBE.map((p) => mv(mat, p));
  const setC = useCallback((r: number, c: number, v: string) => {
    const n = Number(v);
    if (Number.isNaN(n)) return;
    setMat((m) => {
      const cp = m.map((row) => [...row]);
      cp[r][c] = n;
      return cp;
    });
    setPreset('custom');
  }, []);
  const apply = (key: string) => {
    const p = PS.find((x) => x.k === key);
    if (!p) return;
    setMat(p.m.map((r) => [...r]));
    setPreset(key);
  };
  const p = CUBE[sel],
    q = tf[sel];
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-8">
      <h2 className="text-lg font-semibold mb-1">{t('matrix3d.title')}</h2>
      <p className="text-sm text-slate-600 mb-4">{t('matrix3d.subtitle')}</p>
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          {/* biome-ignore lint/a11y/noStaticElementInteractions: custom drag canvas */}
          <div
            className="relative w-full max-w-[320px] h-[300px] mx-auto cursor-grab active:cursor-grabbing select-none"
            onMouseDown={(e) => {
              drag.current = { on: true, lx: e.clientX, ly: e.clientY };
            }}
            onMouseMove={(e) => {
              if (!drag.current.on) return;
              const dy = e.clientY - drag.current.ly;
              drag.current.ly = e.clientY;
              setRx((x) => x - dy * 0.5);
            }}
            onMouseUp={() => {
              drag.current.on = false;
            }}
            onMouseLeave={() => {
              drag.current.on = false;
            }}
          >
            <svg
              viewBox="0 0 280 280"
              className="w-full h-full"
              role="img"
              aria-label={t('matrix3d.svgLabel')}
            >
              <title>{t('matrix3d.svgTitle')}</title>
              <defs>
                <marker
                  id="ax"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M0 0L10 5L0 10z" fill="#ef4444" />
                </marker>
                <marker
                  id="ay"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M0 0L10 5L0 10z" fill="#22c55e" />
                </marker>
                <marker
                  id="az"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M0 0L10 5L0 10z" fill="#3b82f6" />
                </marker>
              </defs>
              <g transform={`rotate(${rx},${OY},${OY})`}>
                {EDGES.map(([a, b]) => {
                  const u = proj(tf[a]),
                    v = proj(tf[b]);
                  return (
                    <line
                      key={`edge-${a}-${b}`}
                      x1={u.x}
                      y1={u.y}
                      x2={v.x}
                      y2={v.y}
                      stroke="#94a3b8"
                      strokeWidth={2}
                    />
                  );
                })}
                {CUBE.map((pt, i) => {
                  const o = proj(pt),
                    n = proj(tf[i]);
                  const active = i === sel;
                  return (
                    <g key={`cube-${pt.x}-${pt.y}-${pt.z}`}>
                      <line
                        x1={o.x}
                        y1={o.y}
                        x2={n.x}
                        y2={n.y}
                        stroke={active ? '#f59e0b' : '#cbd5e1'}
                        strokeWidth={active ? 2 : 1}
                        strokeDasharray={active ? undefined : '4 2'}
                      />
                      {/* biome-ignore lint/a11y/noStaticElementInteractions: SVG interactive point */}
                      <circle
                        cx={o.x}
                        cy={o.y}
                        r={active ? 5 : 3}
                        fill="#cbd5e1"
                        className="cursor-pointer"
                        onClick={() => setSel(i)}
                      />
                      {/* biome-ignore lint/a11y/noStaticElementInteractions: SVG interactive point */}
                      <circle
                        cx={n.x}
                        cy={n.y}
                        r={active ? 6 : 4}
                        fill={active ? '#2563eb' : '#64748b'}
                        className="cursor-pointer"
                        onClick={() => setSel(i)}
                      />
                      {active && (
                        <text
                          x={n.x + 8}
                          y={n.y - 8}
                          fill="#2563eb"
                          fontSize={10}
                          fontFamily="monospace"
                        >
                          ({q.x.toFixed(1)},{q.y.toFixed(1)},{q.z.toFixed(1)})
                        </text>
                      )}
                    </g>
                  );
                })}
                <line
                  x1={OX}
                  y1={OY}
                  x2={OX + 90}
                  y2={OY}
                  stroke="#ef4444"
                  markerEnd="url(#ax)"
                />
                <text x={OX + 95} y={OY + 4} fill="#ef4444" fontSize={10}>
                  X
                </text>
                <line
                  x1={OX}
                  y1={OY}
                  x2={OX}
                  y2={OY - 90}
                  stroke="#22c55e"
                  markerEnd="url(#ay)"
                />
                <text x={OX - 4} y={OY - 95} fill="#22c55e" fontSize={10}>
                  Y
                </text>
                <line
                  x1={OX}
                  y1={OY}
                  x2={OX - 60}
                  y2={OY + 60}
                  stroke="#3b82f6"
                  markerEnd="url(#az)"
                />
                <text x={OX - 65} y={OY + 70} fill="#3b82f6" fontSize={10}>
                  Z
                </text>
              </g>
            </svg>
            <p className="text-center text-xs text-slate-400 mt-1">
              {t('matrix3d.dragHint')}
            </p>
          </div>
        </div>
        <div className="flex-1 space-y-4">
          <div className="flex flex-wrap gap-2">
            {PS.map((p) => (
              <button
                key={p.k}
                type="button"
                onClick={() => apply(p.k)}
                className={`px-3 py-1 rounded text-xs font-medium transition ${preset === p.k ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
              >
                {p.l}
              </button>
            ))}
          </div>
          <div className="bg-slate-50 rounded-lg p-3">
            <p className="text-xs font-medium text-slate-700 mb-2">
              3×3 {t('matrix3d.matrixLabel')}
            </p>
            <div className="grid grid-cols-3 gap-2 max-w-[180px]">
              <input
                type="number"
                step="0.1"
                value={mat[0][0]}
                onChange={(e) => setC(0, 0, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[0][1]}
                onChange={(e) => setC(0, 1, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[0][2]}
                onChange={(e) => setC(0, 2, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[1][0]}
                onChange={(e) => setC(1, 0, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[1][1]}
                onChange={(e) => setC(1, 1, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[1][2]}
                onChange={(e) => setC(1, 2, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[2][0]}
                onChange={(e) => setC(2, 0, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[2][1]}
                onChange={(e) => setC(2, 1, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
              <input
                type="number"
                step="0.1"
                value={mat[2][2]}
                onChange={(e) => setC(2, 2, e.target.value)}
                className="w-full px-2 py-1 text-sm font-mono border border-slate-200 rounded text-center focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
            <p className="text-xs font-medium text-blue-900 mb-1">
              {t('matrix3d.formulaTitle')} (point {sel})
            </p>
            <p className="text-xs text-blue-800 font-mono break-all">
              [{mat[0][0]} {mat[0][1]} {mat[0][2]}; {mat[1][0]} {mat[1][1]}{' '}
              {mat[1][2]}; {mat[2][0]} {mat[2][1]} {mat[2][2]}] · [{p.x} {p.y}{' '}
              {p.z}] = [{q.x.toFixed(2)} {q.y.toFixed(2)} {q.z.toFixed(2)}]
            </p>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            {t('matrix3d.explanation')}
          </p>
        </div>
      </div>
    </div>
  );
}
