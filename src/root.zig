//! Neural Network from Scratch in Zig
const std = @import("std");

/// A simple dense matrix stored in row-major order.
pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .rows = rows, .cols = cols, .data = data };
    }

    pub fn deinit(self: Matrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn get(self: Matrix, r: usize, c: usize) f32 {
        std.debug.assert(r < self.rows and c < self.cols);
        return self.data[r * self.cols + c];
    }

    pub fn set(self: *Matrix, r: usize, c: usize, value: f32) void {
        std.debug.assert(r < self.rows and c < self.cols);
        self.data[r * self.cols + c] = value;
    }

    pub fn randomize(self: *Matrix, rng: std.Random, magnitude: f32) void {
        for (self.data) |*v| {
            v.* = (rng.float(f32) - 0.5) * 2.0 * magnitude;
        }
    }

    pub fn dot(out: *Matrix, a: Matrix, b: Matrix) void {
        std.debug.assert(a.cols == b.rows);
        std.debug.assert(out.rows == a.rows);
        std.debug.assert(out.cols == b.cols);
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0;
                for (0..a.cols) |k| {
                    sum += a.get(i, k) * b.get(k, j);
                }
                out.set(i, j, sum);
            }
        }
    }

    pub fn add(out: *Matrix, a: Matrix, b: Matrix) void {
        std.debug.assert(a.rows == b.rows and a.cols == b.cols);
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        for (0..a.data.len) |i| {
            out.data[i] = a.data[i] + b.data[i];
        }
    }

    pub fn transpose(out: *Matrix, a: Matrix) void {
        std.debug.assert(out.rows == a.cols and out.cols == a.rows);
        for (0..a.rows) |i| {
            for (0..a.cols) |j| {
                out.set(j, i, a.get(i, j));
            }
        }
    }

    pub fn apply(out: *Matrix, a: Matrix, func: *const fn (f32) f32) void {
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        for (0..a.data.len) |i| {
            out.data[i] = func(a.data[i]);
        }
    }

    pub fn applyEach(out: *Matrix, a: Matrix, func: *const fn (f32) f32) void {
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        for (0..a.data.len) |i| {
            out.data[i] = func(a.data[i]);
        }
    }

    pub fn copy(out: *Matrix, a: Matrix) void {
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        @memcpy(out.data, a.data);
    }

    pub fn elementWiseMul(out: *Matrix, a: Matrix, b: Matrix) void {
        std.debug.assert(a.rows == b.rows and a.cols == b.cols);
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        for (0..a.data.len) |i| {
            out.data[i] = a.data[i] * b.data[i];
        }
    }

    pub fn scale(out: *Matrix, a: Matrix, s: f32) void {
        std.debug.assert(out.rows == a.rows and out.cols == a.cols);
        for (0..a.data.len) |i| {
            out.data[i] = a.data[i] * s;
        }
    }

    pub fn print(self: Matrix, writer: *std.Io.Writer) !void {
        for (0..self.rows) |r| {
            for (0..self.cols) |c| {
                try writer.print("{d:8.4} ", .{self.get(r, c)});
            }
            try writer.print("\n", .{});
        }
    }
};

/// Activation functions and their derivatives.
pub fn relu(x: f32) f32 {
    return if (x > 0) x else 0.01 * x;
}

pub fn relu_derivative(x: f32) f32 {
    return if (x > 0) 1 else 0.01;
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn sigmoid_derivative(x: f32) f32 {
    const s = sigmoid(x);
    return s * (1.0 - s);
}

/// A dense layer with weights, biases, and cached pre-activations for backprop.
pub const Layer = struct {
    weights: Matrix,
    biases: Matrix,
    pre_activation: Matrix,
    activation: Matrix,
    weight_gradients: Matrix,
    bias_gradients: Matrix,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !Layer {
        return .{
            .weights = try Matrix.init(allocator, output_size, input_size),
            .biases = try Matrix.init(allocator, output_size, 1),
            .pre_activation = try Matrix.init(allocator, output_size, 1),
            .activation = try Matrix.init(allocator, output_size, 1),
            .weight_gradients = try Matrix.init(allocator, output_size, input_size),
            .bias_gradients = try Matrix.init(allocator, output_size, 1),
        };
    }

    pub fn deinit(self: Layer, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
        self.biases.deinit(allocator);
        self.pre_activation.deinit(allocator);
        self.activation.deinit(allocator);
        self.weight_gradients.deinit(allocator);
        self.bias_gradients.deinit(allocator);
    }

    pub fn randomize(self: *Layer, rng: std.Random) void {
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(self.weights.cols)));
        self.weights.randomize(rng, scale);
        self.biases.randomize(rng, scale);
    }

    pub fn forward(self: *Layer, input: Matrix) void {
        Matrix.dot(&self.pre_activation, self.weights, input);
        Matrix.add(&self.pre_activation, self.pre_activation, self.biases);
        Matrix.apply(&self.activation, self.pre_activation, relu);
    }

    pub fn forwardWithActivation(self: *Layer, input: Matrix, act_fn: *const fn (f32) f32) void {
        Matrix.dot(&self.pre_activation, self.weights, input);
        Matrix.add(&self.pre_activation, self.pre_activation, self.biases);
        Matrix.apply(&self.activation, self.pre_activation, act_fn);
    }
};

/// A simple feedforward neural network.
pub const NeuralNetwork = struct {
    layers: []Layer,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, layer_sizes: []const usize) !NeuralNetwork {
        const layers = try allocator.alloc(Layer, layer_sizes.len - 1);
        errdefer allocator.free(layers);
        for (0..layers.len) |i| {
            layers[i] = try Layer.init(allocator, layer_sizes[i], layer_sizes[i + 1]);
        }
        return .{ .layers = layers, .allocator = allocator };
    }

    pub fn deinit(self: NeuralNetwork) void {
        for (self.layers) |layer| {
            layer.deinit(self.allocator);
        }
        self.allocator.free(self.layers);
    }

    pub fn randomize(self: *NeuralNetwork, rng: std.Random) void {
        for (self.layers) |*layer| {
            layer.randomize(rng);
        }
    }

    pub fn predict(self: *NeuralNetwork, input: Matrix) Matrix {
        var current = input;
        for (0..self.layers.len) |i| {
            const act_fn: *const fn (f32) f32 = if (i == self.layers.len - 1) &sigmoid else &relu;
            self.layers[i].forwardWithActivation(current, act_fn);
            current = self.layers[i].activation;
        }
        return current;
    }

    pub fn train(self: *NeuralNetwork, inputs: []const Matrix, targets: []const Matrix, epochs: usize, learning_rate: f32) !void {
        std.debug.assert(inputs.len == targets.len);

        for (0..epochs) |_| {
            for (inputs, targets) |input, target| {
                // Forward pass
                var current = input;
                for (0..self.layers.len) |i| {
                    const act_fn: *const fn (f32) f32 = if (i == self.layers.len - 1) &sigmoid else &relu;
                    self.layers[i].forwardWithActivation(current, act_fn);
                    current = self.layers[i].activation;
                }

                // Backward pass: compute errors for all layers
                const output_layer = &self.layers[self.layers.len - 1];

                // Output error: (output - target) * sigmoid'(pre_activation)
                Matrix.copy(&output_layer.bias_gradients, output_layer.activation);
                for (0..output_layer.bias_gradients.data.len) |i| {
                    output_layer.bias_gradients.data[i] -= target.data[i];
                }
                for (0..output_layer.bias_gradients.data.len) |i| {
                    output_layer.bias_gradients.data[i] *= sigmoid_derivative(output_layer.pre_activation.data[i]);
                }

                // Hidden layers
                if (self.layers.len > 1) {
                    var next_error: Matrix = output_layer.bias_gradients;
                    var next_weights: Matrix = output_layer.weights;

                    var li: usize = self.layers.len - 1;
                    while (li > 0) {
                        li -= 1;
                        const layer = &self.layers[li];

                        var w_t = try Matrix.init(self.allocator, next_weights.cols, next_weights.rows);
                        defer w_t.deinit(self.allocator);
                        Matrix.transpose(&w_t, next_weights);

                        var error_before_act = try Matrix.init(self.allocator, w_t.rows, next_error.cols);
                        defer error_before_act.deinit(self.allocator);
                        Matrix.dot(&error_before_act, w_t, next_error);

                        Matrix.copy(&layer.bias_gradients, error_before_act);
                        for (0..layer.bias_gradients.data.len) |i| {
                            layer.bias_gradients.data[i] *= relu_derivative(layer.pre_activation.data[i]);
                        }

                        next_error = layer.bias_gradients;
                        next_weights = layer.weights;
                    }
                }

                // Update all weights and biases
                var prev_act = input;
                for (self.layers) |*layer| {
                    var p_t = try Matrix.init(self.allocator, prev_act.cols, prev_act.rows);
                    defer p_t.deinit(self.allocator);
                    Matrix.transpose(&p_t, prev_act);
                    Matrix.dot(&layer.weight_gradients, layer.bias_gradients, p_t);

                    for (0..layer.weights.data.len) |i| {
                        layer.weights.data[i] -= learning_rate * layer.weight_gradients.data[i];
                    }
                    for (0..layer.biases.data.len) |i| {
                        layer.biases.data[i] -= learning_rate * layer.bias_gradients.data[i];
                    }

                    prev_act = layer.activation;
                }
            }
        }
    }
};

test "Matrix operations" {
    const allocator = std.testing.allocator;
    var a = try Matrix.init(allocator, 2, 3);
    defer a.deinit(allocator);
    a.set(0, 0, 1); a.set(0, 1, 2); a.set(0, 2, 3);
    a.set(1, 0, 4); a.set(1, 1, 5); a.set(1, 2, 6);

    var b = try Matrix.init(allocator, 3, 2);
    defer b.deinit(allocator);
    b.set(0, 0, 7); b.set(0, 1, 8);
    b.set(1, 0, 9); b.set(1, 1, 10);
    b.set(2, 0, 11); b.set(2, 1, 12);

    var c = try Matrix.init(allocator, 2, 2);
    defer c.deinit(allocator);
    Matrix.dot(&c, a, b);

    try std.testing.expectApproxEqAbs(c.get(0, 0), 58, 0.001);
    try std.testing.expectApproxEqAbs(c.get(0, 1), 64, 0.001);
    try std.testing.expectApproxEqAbs(c.get(1, 0), 139, 0.001);
    try std.testing.expectApproxEqAbs(c.get(1, 1), 154, 0.001);
}

test "NeuralNetwork XOR" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 2, 4, 1 };
    var nn = try NeuralNetwork.init(allocator, &layer_sizes);
    defer nn.deinit();
    nn.randomize(rng);

    var x0 = try Matrix.init(allocator, 2, 1);
    defer x0.deinit(allocator);
    x0.set(0, 0, 0); x0.set(1, 0, 0);

    var x1 = try Matrix.init(allocator, 2, 1);
    defer x1.deinit(allocator);
    x1.set(0, 0, 0); x1.set(1, 0, 1);

    var x2 = try Matrix.init(allocator, 2, 1);
    defer x2.deinit(allocator);
    x2.set(0, 0, 1); x2.set(1, 0, 0);

    var x3 = try Matrix.init(allocator, 2, 1);
    defer x3.deinit(allocator);
    x3.set(0, 0, 1); x3.set(1, 0, 1);

    var y0 = try Matrix.init(allocator, 1, 1);
    defer y0.deinit(allocator);
    y0.set(0, 0, 0);

    var y1 = try Matrix.init(allocator, 1, 1);
    defer y1.deinit(allocator);
    y1.set(0, 0, 1);

    var y2 = try Matrix.init(allocator, 1, 1);
    defer y2.deinit(allocator);
    y2.set(0, 0, 1);

    var y3 = try Matrix.init(allocator, 1, 1);
    defer y3.deinit(allocator);
    y3.set(0, 0, 0);

    const inputs = [_]Matrix{ x0, x1, x2, x3 };
    const targets = [_]Matrix{ y0, y1, y2, y3 };

    try nn.train(&inputs, &targets, 10000, 0.1);

    var pred = nn.predict(x0);
    try std.testing.expect(pred.get(0, 0) < 0.3);
    pred = nn.predict(x1);
    try std.testing.expect(pred.get(0, 0) > 0.7);
    pred = nn.predict(x2);
    try std.testing.expect(pred.get(0, 0) > 0.7);
    pred = nn.predict(x3);
    try std.testing.expect(pred.get(0, 0) < 0.3);
}
