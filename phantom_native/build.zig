// phantom_native/build.zig
// Zig build configuration for the native performance-critical kernels.
//
// Python is the reference / verification layer; Zig provides:
//   - SovereignTensor → Vectorized loops (SIMD)
//   - NeuroCore → Custom resonance kernels
//   - Bind via C ABI (compatible with CFFI / ctypes)
//
// Full dry-compile: zig build -Doptimize=ReleaseFast → attestable binary

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library: core tensor + resonance kernels
    const lib = b.addStaticLibrary(.{
        .name = "phantom_native",
        .root_source_file = b.path("src/neurocore.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Executable: swarm runtime entry point
    const exe = b.addExecutable(.{
        .name = "phantom_swarm",
        .root_source_file = b.path("src/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/neurocore.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
