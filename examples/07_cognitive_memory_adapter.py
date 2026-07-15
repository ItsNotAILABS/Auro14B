"""Example 07: Use cognitive memory adapter for agent integration."""

from mesie.cognitive import SpectralMemoryAdapter, SpectralAnomalyAdapter

# Create a memory adapter
adapter = SpectralMemoryAdapter()

# Convert a spectral record to a memory object
record = {
    "record_id": "sensor_reading_001",
    "components": [
        {
            "name": "accelerometer",
            "frequency": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "amplitude": [0.01, 0.05, 0.12, 0.25, 0.18, 0.08, 0.03, 0.01],
        }
    ],
}

memory = adapter.to_memory_object(record)
print("Memory Object:")
print(f"  semantic_id: {memory['semantic_id']}")
print(f"  embedding length: {len(memory['spectral_embedding'])}")
print(f"  resonance_signature: {memory['resonance_signature']}")
print(f"  coherence: {memory['coherence_signature']:.4f}")
print(f"  confidence: {memory['confidence']}")
print(f"  anomaly_score: {memory['anomaly_score']}")

# Anomaly detection
print("\n--- Anomaly Detection ---")
anomaly_adapter = SpectralAnomalyAdapter(threshold=2.0)

# Fit baseline with normal records
normal_records = [
    {"record_id": f"normal_{i}", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": [0.2 + i*0.01, 0.5, 0.3, 0.1]}]}
    for i in range(5)
]
anomaly_adapter.fit_baseline(normal_records)

# Test normal record
normal_test = {"record_id": "test_normal", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": [0.22, 0.5, 0.3, 0.1]}]}
print(f"Normal record anomaly score: {anomaly_adapter.score_anomaly(normal_test):.4f}")
print(f"Is anomaly: {anomaly_adapter.is_anomaly(normal_test)}")

# Test anomalous record
anomalous = {"record_id": "test_anomaly", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": [5.0, 5.0, 5.0, 5.0]}]}
print(f"Anomalous record score: {anomaly_adapter.score_anomaly(anomalous):.4f}")
print(f"Is anomaly: {anomaly_adapter.is_anomaly(anomalous)}")
