#!/usr/bin/env python3
"""Example 08: 3D Connectome Brain — The Intelligence Backend.

Demonstrates the simulated 3D brain environment where the connectome
IS the backend, IS the AI, IS the intelligence.

This example:
1. Builds the full 3D connectome with real brain regions
2. Injects stimuli to simulate sensory input
3. Runs the neural propagation simulation
4. Shows how activation spreads through brain systems
5. Reports emergent dynamics (coherence, dominant system)
"""

from mesie.connectome import (
    BrainSystem,
    ConnectomeEnvironment3D,
    build_default_connectome,
    get_default_regions,
    get_regions_by_system,
)


def main() -> None:
    """Run the 3D connectome brain simulation."""
    print("=" * 70)
    print("  MESIE — 3D Connectome Intelligence Backend")
    print("  The Brain IS the Backend IS the AI IS the Intelligence")
    print("=" * 70)
    print()

    # --- 1. Build the Connectome ---
    connectome = build_default_connectome()
    print(f"[CONNECTOME] Built with {connectome.num_regions} brain regions "
          f"and {connectome.num_connections} connections")
    print()

    # Show brain systems
    print("  Brain Systems (Real Anatomical Regions):")
    for system in BrainSystem:
        regions = get_regions_by_system(system)
        names = [r.abbreviation for r in regions]
        print(f"    {system.value:14s} → {len(regions)} regions: {', '.join(names)}")
    print()

    # --- 2. Create the 3D Environment ---
    env = ConnectomeEnvironment3D(
        connectome=connectome,
        dt_ms=1.0,
        decay_rate=0.03,
        propagation_gain=0.7,
        noise_level=0.005,
    )
    print(f"[ENVIRONMENT] 3D simulation initialized")
    print(f"  Timestep: {env.dt_ms} ms")
    print(f"  Regions: {env.num_regions}")
    print()

    # --- 3. Simulate Sensory Input ---
    print("[STIMULUS] Injecting visual input into V1 (Primary Visual Cortex)...")
    env.inject_stimulus("V1", amplitude=0.9, frequency_hz=40.0)
    print("[STIMULUS] Injecting language input into WER (Wernicke's Area)...")
    env.inject_stimulus("WER", amplitude=0.8, frequency_hz=30.0)
    print("[STIMULUS] Injecting executive signal to DLPFC_L...")
    env.inject_stimulus("DLPFC_L", amplitude=0.7, frequency_hz=45.0)
    print()

    # --- 4. Run Simulation ---
    print("[SIMULATION] Running 50ms of neural dynamics...")
    states = env.run(duration_ms=50.0)
    print(f"  Computed {len(states)} timesteps")
    print()

    # --- 5. Report Results ---
    final_state = states[-1]
    print("[RESULTS] Final state at t = {:.1f} ms:".format(final_state.timestamp))
    print(f"  Global Coherence: {final_state.global_coherence:.4f}")
    print(f"  Dominant System:  {final_state.dominant_system.value}")
    print()

    # Show per-system activation
    print("  System Activations:")
    for system in BrainSystem:
        act = env.get_system_activation(system)
        bar = "█" * int(act * 40)
        print(f"    {system.value:14s} [{act:.3f}] {bar}")
    print()

    # Show top-10 most active regions
    activations = env.get_all_activations()
    sorted_regions = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    print("  Top 10 Active Regions:")
    for abbr, act in sorted_regions[:10]:
        region = connectome.get_region(abbr)
        pos = region.position_3d if region else (0, 0, 0)
        print(f"    {abbr:8s} → {act:.4f}  pos=({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f})")
    print()

    # --- 6. 3D State Export ---
    state_3d = env.get_3d_state()
    print("[3D STATE] Exportable environment snapshot:")
    print(f"  Positions shape: {state_3d['positions'].shape}")
    print(f"  Active edges:    {len(state_3d['edges'])}")
    print(f"  Coherence:       {state_3d['global_coherence']:.4f}")
    print(f"  Dominant:        {state_3d['dominant_system']}")
    print()

    # --- 7. Distance Matrix ---
    dist_matrix = connectome.build_distance_matrix()
    print(f"[TOPOLOGY] Distance matrix shape: {dist_matrix.shape}")
    print(f"  Mean inter-region distance: {dist_matrix[dist_matrix > 0].mean():.1f} mm")
    print(f"  Max distance:               {dist_matrix.max():.1f} mm")
    print()

    print("=" * 70)
    print("  Simulation complete. The connectome brain backend is alive.")
    print("=" * 70)


if __name__ == "__main__":
    main()
