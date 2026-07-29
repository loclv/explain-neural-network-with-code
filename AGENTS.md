# AGENTS.md

Two-project monorepo. Zig NN engine (root) + React/TS visualizer (`explain-neural-network/`).

## Zig package (root)

- `src/root.zig` — library: Matrix, Layer, NeuralNetwork, activations, SGD with momentum
- `src/main.zig` — CLI demo (10×10 grid point counter, 100-64-1 network)
- `zig build run` — run CLI demo
- `zig build test` — run all tests (library + exe)
- Requires Zig 0.16.0
- Uses arena allocator; `Matrix.deinit()` frees data slice
- Network topology set at init; last layer sigmoid, hidden layers Leaky ReLU
- Training: `net.train(inputs, targets, epochs, learningRate, momentum)`

## Web visualizer (`explain-neural-network/`)

- Rsbuild + React 19 + Tailwind CSS 4 + TypeScript + Biome
- `bun run dev` — dev server at http://localhost:3000
- `bun run build` — production bundle in `dist/`
- `bun run format` — Biome format (spaces, single quotes)
- `bun run check` — Biome lint + format + organize imports
- `bun run test` — Rstest (Vitest-compatible, happy-dom)
- `bun run d` — `rsbuild --open` (opens browser)
- tsconfig: `verbatimModuleSyntax` (use `import type`), `noUnusedLocals`, `noUnusedParameters`
- Biome: `indentStyle: space`, `quoteStyle: single`, VCS mode uses `.gitignore`
- Entry: `src/index.tsx` → `src/App.tsx`
- i18n: react-i18next with `src/i18n.ts` (en/vi locales)
- `src/nn-engine.ts` — JS port of the Zig NN; same algorithm, no dependencies