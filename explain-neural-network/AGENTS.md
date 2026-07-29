# AGENTS.md

React 19 + Rsbuild + Tailwind CSS 4 + TypeScript + Biome visualizer for a neural network from scratch.

## Commands

- `bun run dev` — dev server at http://localhost:3000
- `bun run build` — production bundle in `dist/`
- `bun run format` — Biome format (spaces, single quotes)
- `bun run check` — Biome lint + format + organize imports
- `bun run test` — Rstest (happy-dom)
- `bun run test:watch` — Rstest watch mode
- `bun run d` — `rsbuild --open`

## Conventions

- `verbatimModuleSyntax` (use `import type` for type-only imports)
- `noUnusedLocals`, `noUnusedParameters` enabled
- Biome: `indentStyle: space`, `quoteStyle: single`, VCS mode uses `.gitignore`
- i18n: react-i18next, translations in `src/i18n.ts` (en/vi)
- Entry: `src/index.tsx` → `src/App.tsx`
- Neural engine (no-dependency JS port of Zig NN): `src/nn-engine.ts`

## Docs

- Rsbuild: https://rsbuild.rs/llms.txt
- Rstest: https://rstest.rs/llms.txt