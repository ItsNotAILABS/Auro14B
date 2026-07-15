// neurocore.zig — Sovereign NeuroCore SIMD Kernels
//
// Production-grade native kernels for the Phantom-MESIE stack.
// Uses explicit SIMD intrinsics for vectorized spectral operations.
// C ABI exports for Python/CFFI binding.

const std = @import("std");
const math = std.math;

// ============================================================
// SovereignTensor — Cache-friendly f32 buffer with metadata
// ============================================================

pub const SpectralMeta = struct {
    resonance: f32 = 1.0,
    helix_turns: u32 = 8,
    quantized: bool = false,
    quant_scale: f32 = 1.0,
};

pub const SovereignTensor = struct {
    data: []f32,
    shape: []const usize,
    meta: SpectralMeta,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, meta: SpectralMeta) !SovereignTensor {
        var size: usize = 1;
        for (shape) |d| {
            size *= d;
        }
        const data = try allocator.alloc(f32, size);
        @memset(data, 0.0);
        return SovereignTensor{
            .data = data,
            .shape = shape,
            .meta = meta,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SovereignTensor) void {
        self.allocator.free(self.data);
    }

    pub fn size(self: SovereignTensor) usize {
        return self.data.len;
    }
};

// ============================================================
// SIMD Vectorized Operations
// ============================================================

/// 8-wide SIMD vector addition with scalar tail handling.
/// Resonance product is applied to the result.
pub fn vector_add_simd(a: SovereignTensor, b: SovereignTensor, out: []f32) void {
    const n = @min(a.data.len, b.data.len);
    const resonance = a.meta.resonance * b.meta.resonance;

    var i: usize = 0;

    // 8-wide SIMD loop
    while (i + 8 <= n) : (i += 8) {
        const va: @Vector(8, f32) = a.data[i..][0..8].*;
        const vb: @Vector(8, f32) = b.data[i..][0..8].*;
        const vr = va + vb;
        const res_vec: @Vector(8, f32) = @splat(resonance);
        const scaled = vr * res_vec;
        out[i..][0..8].* = scaled;
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        out[i] = (a.data[i] + b.data[i]) * resonance;
    }
}

/// 8-wide SIMD element-wise multiplication.
pub fn vector_mul_simd(a: SovereignTensor, b: SovereignTensor, out: []f32) void {
    const n = @min(a.data.len, b.data.len);

    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const va: @Vector(8, f32) = a.data[i..][0..8].*;
        const vb: @Vector(8, f32) = b.data[i..][0..8].*;
        out[i..][0..8].* = va * vb;
    }
    while (i < n) : (i += 1) {
        out[i] = a.data[i] * b.data[i];
    }
}

/// Resonance-weighted matrix multiplication with 4-wide inner unrolling.
/// For fixed spectral shapes (e.g., 128x128, 64x64).
pub fn resonance_matmul(
    a: SovereignTensor,
    b: SovereignTensor,
    out: []f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    const resonance = a.meta.resonance * b.meta.resonance;

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;
            var p: usize = 0;

            // 4-wide inner unroll
            while (p + 4 <= k) : (p += 4) {
                acc += a.data[i * k + p] * b.data[p * n + j] +
                    a.data[i * k + p + 1] * b.data[(p + 1) * n + j] +
                    a.data[i * k + p + 2] * b.data[(p + 2) * n + j] +
                    a.data[i * k + p + 3] * b.data[(p + 3) * n + j];
            }
            // Scalar tail
            while (p < k) : (p += 1) {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }

            out[i * n + j] = acc * resonance;
        }
    }
}

// ============================================================
// Resonance Attention Kernel
// ============================================================

/// Native resonance attention: dot-product with exponential decay + softmax.
pub fn resonance_attention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,
    n: usize,
) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));

    // Compute scores with resonance decay
    var scores: [512]f32 = undefined;
    var max_score: f32 = -math.inf(f32);

    for (0..n) |i| {
        const dot = q[i] * k[i] * scale;
        const resonance = @exp(-@abs(dot) * 0.5);
        scores[i] = dot * resonance;
        if (scores[i] > max_score) max_score = scores[i];
    }

    // Stable softmax
    var total: f32 = 0.0;
    for (0..n) |i| {
        scores[i] = @exp(scores[i] - max_score);
        total += scores[i];
    }
    if (total == 0.0) total = 1.0;

    // Weighted output
    for (0..n) |i| {
        out[i] = (scores[i] / total) * v[i];
    }
}

// ============================================================
// Helix Encoding
// ============================================================

/// Apply helix rotation encoding for spectral retrieval.
pub fn helix_encode(data: []const f32, out: []f32, n: usize, turns: u32) void {
    const n_f: f32 = @floatFromInt(if (n > 1) n - 1 else 1);
    const turns_f: f32 = @floatFromInt(turns);

    for (0..n) |i| {
        const i_f: f32 = @floatFromInt(i);
        const phase = 2.0 * math.pi * turns_f * (i_f / n_f);
        out[i] = data[i] * @cos(phase) + @sin(phase) * 0.1;
    }
}

// ============================================================
// INT8 Quantization for Edge Deployment
// ============================================================

/// Quantize f32 tensor to INT8 range [-127, 127].
pub fn quantize_int8(data: []const f32, out: []i8, n: usize) f32 {
    var max_abs: f32 = 0.0;
    for (0..n) |i| {
        const abs_val = @abs(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0) max_abs = 1.0;

    for (0..n) |i| {
        const scaled = (data[i] / max_abs) * 127.0;
        out[i] = @intFromFloat(@min(@max(scaled, -127.0), 127.0));
    }

    return max_abs; // return scale for dequantization
}

// ============================================================
// C ABI Exports (for Python CFFI / ctypes binding)
// ============================================================

export fn phantom_vector_add(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, f32) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        out[i..][0..8].* = va + vb;
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] + b[i];
    }
}

export fn phantom_resonance_matmul(
    a: [*]const f32,
    b: [*]const f32,
    out: [*]f32,
    m: u32,
    k: u32,
    n: u32,
    resonance: f32,
) void {
    const m_sz = @as(usize, m);
    const k_sz = @as(usize, k);
    const n_sz = @as(usize, n);

    for (0..m_sz) |i| {
        for (0..n_sz) |j| {
            var acc: f32 = 0.0;
            for (0..k_sz) |p| {
                acc += a[i * k_sz + p] * b[p * n_sz + j];
            }
            out[i * n_sz + j] = acc * resonance;
        }
    }
}

export fn phantom_helix_encode(data: [*]const f32, out: [*]f32, n: u32, turns: u32) void {
    const len = @as(usize, n);
    const n_f: f32 = @floatFromInt(if (len > 1) len - 1 else 1);
    const turns_f: f32 = @floatFromInt(turns);

    for (0..len) |i| {
        const i_f: f32 = @floatFromInt(i);
        const phase = 2.0 * math.pi * turns_f * (i_f / n_f);
        out[i] = data[i] * @cos(phase) + @sin(phase) * 0.1;
    }
}

// ============================================================
// Tests
// ============================================================

test "vector_add_simd basic" {
    const allocator = std.testing.allocator;
    var a_data = try allocator.alloc(f32, 16);
    defer allocator.free(a_data);
    var b_data = try allocator.alloc(f32, 16);
    defer allocator.free(b_data);
    var out = try allocator.alloc(f32, 16);
    defer allocator.free(out);

    for (0..16) |i| {
        a_data[i] = @floatFromInt(i);
        b_data[i] = 1.0;
    }

    const shape = &[_]usize{16};
    const a = SovereignTensor{ .data = a_data, .shape = shape, .meta = .{}, .allocator = allocator };
    const b = SovereignTensor{ .data = b_data, .shape = shape, .meta = .{}, .allocator = allocator };

    vector_add_simd(a, b, out);

    try std.testing.expectApproxEqAbs(out[0], 1.0, 0.001);
    try std.testing.expectApproxEqAbs(out[7], 8.0, 0.001);
    try std.testing.expectApproxEqAbs(out[15], 16.0, 0.001);
}

test "resonance_attention basic" {
    var q = [_]f32{ 1.0, 0.5, 0.3, 0.1 };
    var k_arr = [_]f32{ 0.5, 1.0, 0.2, 0.8 };
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;

    resonance_attention(&q, &k_arr, &v, &out, 4);

    // Output should be non-zero weighted values
    var total: f32 = 0.0;
    for (out[0..4]) |val| {
        total += val;
        try std.testing.expect(val >= 0.0);
    }
    try std.testing.expect(total > 0.0);
}
