/**
 * MatrixMultiplyDemo — child-friendly visual explanation of matrix multiplication.
 *
 * Shows a concrete 2×3 · 3×2 example with colour-coded rows and columns.
 * Pressing "Next step" walks through every dot-product so a child can see
 * *why* each result cell gets its value.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const ROWS_A = 2;
const COLS_A = 3;
const COLS_B = 2;

const A = [
  [2, 1, 3],
  [0, 4, 1],
];
const B = [
  [1, 2],
  [3, 0],
  [1, 1],
];

function computeResult(): number[][] {
  const C: number[][] = [];
  for (let i = 0; i < ROWS_A; i++) {
    C[i] = [];
    for (let j = 0; j < COLS_B; j++) {
      let sum = 0;
      for (let k = 0; k < COLS_A; k++) sum += A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  return C;
}

const RESULT = computeResult();

interface Step {
  i: number;
  j: number;
  k: number;
  partial: number;
  done: boolean;
}

function buildSteps(): Step[] {
  const steps: Step[] = [];
  for (let i = 0; i < ROWS_A; i++) {
    for (let j = 0; j < COLS_B; j++) {
      let partial = 0;
      for (let k = 0; k < COLS_A; k++) {
        partial += A[i][k] * B[k][j];
        steps.push({ i, j, k, partial, done: false });
      }
      steps.push({ i, j, k: COLS_A, partial, done: true });
    }
  }
  return steps;
}

const ALL_STEPS = buildSteps();
const ROW_COLORS = ['bg-sky-100 text-sky-800', 'bg-amber-100 text-amber-800'];
const COL_COLORS = [
  'bg-rose-100 text-rose-800',
  'bg-emerald-100 text-emerald-800',
];

export default function MatrixMultiplyDemo() {
  const { t } = useTranslation();
  const [stepIndex, setStepIndex] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = ALL_STEPS[Math.min(stepIndex, ALL_STEPS.length - 1)];

  const goNext = useCallback(() => {
    setStepIndex((s) => Math.min(s + 1, ALL_STEPS.length));
  }, []);

  const goPrev = useCallback(() => {
    setStepIndex((s) => Math.max(s - 1, 0));
  }, []);

  const reset = useCallback(() => {
    setStepIndex(0);
    setAutoPlay(false);
  }, []);

  useEffect(() => {
    if (autoPlay) {
      timerRef.current = setInterval(() => {
        setStepIndex((s) => {
          if (s >= ALL_STEPS.length) {
            setAutoPlay(false);
            return s;
          }
          return s + 1;
        });
      }, 800);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [autoPlay]);

  const isDone = stepIndex >= ALL_STEPS.length;

  function cellClass(matrix: 'A' | 'B' | 'C', r: number, c: number): string {
    const base =
      'w-12 h-12 flex items-center justify-center rounded-lg font-mono text-lg font-bold transition-all duration-300 border-2 ';
    if (matrix === 'C') {
      if (isDone) return base + 'bg-slate-100 text-slate-700 border-slate-200';
      if (r === step.i && c === step.j && step.done)
        return base + COL_COLORS[c] + ' border-current scale-110 shadow-md';
      if (r === step.i && c === step.j)
        return base + COL_COLORS[c] + ' border-current';
      if (step.done && (r < step.i || (r === step.i && c < step.j)))
        return base + 'bg-slate-100 text-slate-700 border-slate-200';
      return base + 'bg-slate-50 text-slate-300 border-slate-100';
    }
    if (matrix === 'A') {
      const active = !isDone && r === step.i;
      if (active && c <= step.k && !step.done)
        return base + ROW_COLORS[r] + ' border-current scale-105';
      if (active || (isDone && r === step.i))
        return base + ROW_COLORS[r] + ' border-current';
      return base + 'bg-slate-50 text-slate-400 border-slate-100';
    }
    // matrix === 'B'
    const active = !isDone && c === step.j;
    if (active && r <= step.k && !step.done)
      return base + COL_COLORS[c] + ' border-current scale-105';
    if (active || (isDone && c === step.j))
      return base + COL_COLORS[c] + ' border-current';
    return base + 'bg-slate-50 text-slate-400 border-slate-100';
  }

  function showArrowA(r: number, c: number): boolean {
    if (isDone) return false;
    return r === step.i && c === step.k && !step.done;
  }

  function showArrowB(r: number, c: number): boolean {
    if (isDone) return false;
    return c === step.j && r === step.k && !step.done;
  }

  const progress = Math.round((stepIndex / ALL_STEPS.length) * 100);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-8">
      <h2 className="text-lg font-semibold mb-1">{t('matrixDemo.title')}</h2>
      <p className="text-sm text-slate-600 mb-4">{t('matrixDemo.subtitle')}</p>

      {/* Matrices */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-8 mb-6">
        {/* Matrix A */}
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
            {t('matrixDemo.matrixA')}
          </span>
          <div className="relative">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${COLS_A}, minmax(0, 1fr))`,
              }}
            >
              {A.map((row, r) =>
                row.map((v, c) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: static demo grid
                  <div key={`ma-r${r}-c${c}`} className="relative">
                    {showArrowA(r, c) && (
                      <span className="absolute -right-3 top-1/2 -translate-y-1/2 text-sky-600 text-xl animate-pulse">
                        →
                      </span>
                    )}
                    <div className={cellClass('A', r, c)}>{v}</div>
                  </div>
                )),
              )}
            </div>
          </div>
        </div>

        {/* Multiply sign */}
        <div className="text-3xl font-bold text-slate-400">×</div>

        {/* Matrix B */}
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
            {t('matrixDemo.matrixB')}
          </span>
          <div className="relative">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${COLS_B}, minmax(0, 1fr))`,
              }}
            >
              {B.map((row, r) =>
                row.map((v, c) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: static demo grid
                  <div key={`mb-r${r}-c${c}`} className="relative">
                    {showArrowB(r, c) && (
                      <span className="absolute -bottom-3 left-1/2 -translate-x-1/2 text-rose-600 text-xl animate-pulse">
                        ↓
                      </span>
                    )}
                    <div className={cellClass('B', r, c)}>{v}</div>
                  </div>
                )),
              )}
            </div>
          </div>
        </div>

        {/* Equals sign */}
        <div className="text-3xl font-bold text-slate-400">=</div>

        {/* Matrix C */}
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">
            {t('matrixDemo.result')}
          </span>
          <div
            className="grid gap-2"
            style={{ gridTemplateColumns: `repeat(${COLS_B}, minmax(0, 1fr))` }}
          >
            {RESULT.map((row, r) =>
              row.map((v, c) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: static demo grid
                <div key={`mc-r${r}-c${c}`} className={cellClass('C', r, c)}>
                  {isDone || (r === step.i && c === step.j && step.done)
                    ? v
                    : '?'}
                </div>
              )),
            )}
          </div>
        </div>
      </div>

      {/* Formula bar */}
      <div className="bg-slate-50 rounded-lg p-4 mb-4 min-h-[3.5rem] flex items-center justify-center text-center">
        {isDone ? (
          <span className="text-slate-700 font-medium">
            {t('matrixDemo.allDone')}
          </span>
        ) : step.done ? (
          <span className="text-slate-700">
            {t('matrixDemo.cellComplete', {
              row: step.i + 1,
              col: step.j + 1,
              value: RESULT[step.i][step.j],
            })}
          </span>
        ) : (
          <span className="text-slate-700 font-mono text-base">
            {t('matrixDemo.computing', {
              row: step.i + 1,
              col: step.j + 1,
              a: A[step.i][step.k],
              b: B[step.k][step.j],
              partial: step.partial,
            })}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="w-full bg-slate-100 rounded-full h-2.5 mb-4 overflow-hidden">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3 flex-wrap">
        <button
          type="button"
          onClick={goPrev}
          disabled={stepIndex === 0}
          className="px-4 py-2 rounded-lg bg-slate-200 text-slate-700 font-medium hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          {t('matrixDemo.prev')}
        </button>
        <button
          type="button"
          onClick={() => setAutoPlay((p) => !p)}
          className={`px-4 py-2 rounded-lg font-medium transition ${
            autoPlay
              ? 'bg-amber-500 text-white hover:bg-amber-600'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {autoPlay ? t('matrixDemo.pause') : t('matrixDemo.play')}
        </button>
        <button
          type="button"
          onClick={goNext}
          disabled={stepIndex >= ALL_STEPS.length}
          className="px-4 py-2 rounded-lg bg-slate-200 text-slate-700 font-medium hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          {t('matrixDemo.next')}
        </button>
        <button
          type="button"
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-100 text-slate-600 font-medium hover:bg-slate-200 transition"
        >
          {t('matrixDemo.reset')}
        </button>
      </div>
    </div>
  );
}
