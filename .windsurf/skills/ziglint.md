---
description: Follow ziglint rules when writing or editing Zig code
---

When writing or editing Zig source files in this project, ensure the code passes `ziglint` (v0.5.2).

Key rules to follow:

- **Z001 — camelCase functions**: All function names must be `camelCase`.  
  Bad: `relu_derivative`  
  Good: `reluDerivative`

- **Z016 — Split compound asserts**: Never write `assert(a and b)`; split into two separate asserts.  
  Bad: `std.debug.assert(r < self.rows and c < self.cols);`  
  Good:
  ```zig
  std.debug.assert(r < self.rows);
  std.debug.assert(c < self.cols);
  ```

- **Z024 — Max line length**: No line may exceed 120 characters. Break long function signatures, struct literals, and argument lists across multiple lines using trailing commas.

Before finishing any Zig-related task, run `ziglint` and fix all warnings.
