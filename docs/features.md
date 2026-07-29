# Features

## Neural Network Engine (Zig + JS ports)

- **Matrix**: Row-major dense `f32`/`Float32Array` with dot, add, transpose, element-wise multiply, copy, apply
- **Layer**: Dense layer with weights, biases, pre-activation, activation, gradients, momentum velocities — all pre-allocated at init
- **Activations**: Leaky ReLU (hidden, slope 0.01) and sigmoid (output), with analytic derivatives
- **Initialization**: He uniform (`sqrt(2/fan_in)`)
- **Training**: Single-sample SG