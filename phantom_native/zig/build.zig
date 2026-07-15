// build.zig — Sovereign Native Stack build configuration
// Produces attestable static library + swarm runtime executable.
//
// Build commands:
//   zig build                          # Debug build
//   zig build -Doptimize=ReleaseFast   # Production (SIMD-optimized)
//   zig build -Doptimize=ReleaseFast -Dtarget=aarch64-linux  # ARM edge
//
// Output:
//   zig-out/lib/libphantom_native.a    # Static library (C ABI)
//   zig-out/bin/phantom_swarm          # Swarm runtime executable

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library — core kernels (C ABI compatible)
    const lib = b.addStaticLibrary(.{
        .name = "phantom_native",
        .root_source_file = b.path("src/neurocore.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Swarm runtime executable
    const exe = b.addExecutable(.{
        .name = "phantom_swarm",
        .root_source_file = b.path("src/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibrary(lib);
    b.installArtifact(exe);

    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("src/neurocore.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run native kernel tests");
    test_step.dependOn(&run_tests.step);
}
