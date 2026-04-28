const std = @import("std");
const Io = std.Io;
const nn = @import("zig");

const GRID = 10;
const INPUT = GRID * GRID;
const SAMPLES = 200;
const TESTS = 10;

fn makeGrid(allocator: std.mem.Allocator, rng: std.Random, points: usize) !nn.Matrix {
    var m = try nn.Matrix.init(allocator, INPUT, 1);
    var placed: usize = 0;
    while (placed < points) {
        const idx = rng.intRangeLessThan(usize, 0, INPUT);
        if (m.data[idx] == 0) {
            m.data[idx] = 1.0;
            placed += 1;
        }
    }
    return m;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const w = &stdout_file_writer.interface;

    try w.print("10x10 Grid Point Counter\n========================\n\n", .{});

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const sizes = [_]usize{ INPUT, 64, 1 };
    var net = try nn.NeuralNetwork.init(arena, &sizes);
    defer net.deinit();
    net.randomize(rng);

    try w.print("Network: {d}-{d}-{d}\n", .{ sizes[0], sizes[1], sizes[2] });
    try w.print("Generating {d} training samples...\n\n", .{SAMPLES});

    var inputs: std.ArrayList(nn.Matrix) = .empty;
    defer {
        for (inputs.items) |m| m.deinit(arena);
        inputs.deinit(arena);
    }

    var targets: std.ArrayList(nn.Matrix) = .empty;
    defer {
        for (targets.items) |m| m.deinit(arena);
        targets.deinit(arena);
    }

    for (0..SAMPLES) |i| {
        const pts: usize = if (i % 2 == 0) 1 else 2;
        try inputs.append(arena, try makeGrid(arena, rng, pts));
        var t = try nn.Matrix.init(arena, 1, 1);
        t.set(0, 0, if (pts == 1) 0.0 else 1.0);
        try targets.append(arena, t);
    }

    try w.print("Training (1 point = 0.0, 2 points = 1.0)...\n", .{});
    try net.train(inputs.items, targets.items, 500, 0.01);

    try w.print("\nPredictions on new random grids:\n", .{});
    for (0..TESTS) |i| {
        const pts: usize = if (i % 2 == 0) 1 else 2;
        var test_m = try makeGrid(arena, rng, pts);
        defer test_m.deinit(arena);
        const pred = net.predict(test_m);
        const label: f32 = if (pts == 1) 0.0 else 1.0;
        try w.print("  {d} point(s) -> prediction = {d:.4} (target = {d:.1})\n", .{
            pts, pred.get(0, 0), label,
        });
    }

    try w.flush();
}
