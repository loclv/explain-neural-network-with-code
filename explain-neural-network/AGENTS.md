# AGENTS.md

You are an expert in JavaScript, Rsbuild, and web application development. You write maintainable, performant, and accessible code.

## Commands

- `bun run dev` - Start the dev server
- `bun run build` - Build the app for production
- `bun run preview` - Preview the production build locally

## Docs

- Rsbuild: https://rsbuild.rs/llms.txt
- Rspack: https://rspack.rs/llms.txt
- Rstest: https://rstest.rs/llms.txt

## Tools

### Rstest

- Run `bun run test` to run tests
- Run `bun run test:watch` to run tests in watch mode

### Biome

- Run `bun run lint` to lint your code
- Run `bun run format` to format your code

# Token-optimized CLI

Always use `rtk` directly for meta commands.

## Meta Commands (always use rtk directly)

```bash
rtk discover # Analyze Claude Code history for missed opportunities
rtk proxy <cmd> # Execute raw command without filtering (for debugging)
# Files
rtk ls . # Token-optimized directory tree
rtk read file.rs # Smart file reading
rtk read file.rs -l aggressive # Signatures only (strips bodies)
rtk smart file.rs # 2-line heuristic code summary
rtk find "*.rs" . # Compact find results
rtk grep "pattern" . # Grouped search results
rtk diff file1 file2 # Condensed diff
rtk git status # Compact status
rtk git log -n 10 # One-line commits
rtk git diff # Condensed diff
rtk git add # -> "ok"
rtk git commit -m "msg" # -> "ok abc1234"
rtk git push # -> "ok main"
rtk git pull # -> "ok 3 files +10 -2"
rtk npm # for `npm` command, node package manager
rtk bun # for `bun` command, bun package manager
rtk bun list # Compact dependency tree
rtk curl <url> # Truncate + save full output
```

## Hook-Based Usage

All other commands are automatically rewritten by the Claude Code hook.
Example: `git status` → `rtk git status` (transparent, 0 tokens overhead)
