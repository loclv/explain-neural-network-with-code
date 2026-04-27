const std = @import("std");
const Io = std.Io;
const nn = @import("zig");

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try stdout_writer.print("Neural Network from Scratch in Zig\n", .{});
    try stdout_writer.print("====================================\n\n", .{});

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const layer_sizes = [_]usize{ 2, 4, 1 };
    var network = try nn.NeuralNetwork.init(arena, &layer_sizes);
    defer network.deinit();
    network.randomize(rng);

    try stdout_writer.print("Training a {d}-{d}-{d} network on XOR...\n", .{
        layer_sizes[0], layer_sizes[1], layer_sizes[2],
    });

    var x0 = try nn.Matrix.init(arena, 2, 1);
    defer x0.deinit(arena);
    x0.set(0, 0, 0); x0.set(1, 0, 0);

    var x1 = try nn.Matrix.init(arena, 2, 1);
    defer x1.deinit(arena);
    x1.set(0, 0, 0); x1.set(1, 0, 1);

    var x2 = try nn.Matrix.init(arena, 2, 1);
    defer x2.deinit(arena);
    x2.set(0, 0, 1); x2.set(1, 0, 0);

    var x3 = try nn.Matrix.init(arena, 2, 1);
    defer x3.deinit(arena);
    x3.set(0, 0, 1); x3.set(1, 0, 1);

    var y0 = try nn.Matrix.init(arena, 1, 1);
    defer y0.deinit(arena);
    y0.set(0, 0, 0);

    var y1 = try nn.Matrix.init(arena, 1, 1);
    defer y1.deinit(arena);
    y1.set(0, 0, 1);

    var y2 = try nn.Matrix.init(arena, 1, 1);
    defer y2.deinit(arena);
    y2.set(0, 0, 1);

    var y3 = try nn.Matrix.init(arena, 1, 1);
    defer y3.deinit(arena);
    y3.set(0, 0, 0);

    const inputs = [_]nn.Matrix{ x0, x1, x2, x3 };
    const targets = [_]nn.Matrix{ y0, y1, y2, y3 };

    try network.train(&inputs, &targets, 10000, 0.1);

    try stdout_writer.print("\nPredictions after training:\n", .{});
    const labels = [_][]const u8{ "0 XOR 0", "0 XOR 1", "1 XOR 0", "1 XOR 1" };
    const inputs_demo = [_]nn.Matrix{ x0, x1, x2, x3 };

    for (inputs_demo, labels) |inp, label| {
        const pred = network.predict(inp);
        try stdout_writer.print("  {s} = {d:.4}\n", .{ label, pred.get(0, 0) });
    }

    try stdout_writer.flush();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
