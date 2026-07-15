"""Example 06: Create spectral embeddings for AI workflows."""

import numpy as np
from mesie import load_record
from mesie.embeddings import SpectralVectorizer, SpectralRetriever

# Create some sample records
records = [
    {"record_id": "earthquake_1", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.01, 0.08, 0.15, 0.12, 0.04, 0.01]}]},
    {"record_id": "earthquake_2", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.02, 0.09, 0.18, 0.10, 0.03, 0.01]}]},
    {"record_id": "vibration_1", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.001, 0.002, 0.01, 0.5, 0.8, 0.3]}]},
]

# Create embeddings
vectorizer = SpectralVectorizer()
print(f"Embedding dimension: {vectorizer.embedding_dim}")

for record in records:
    emb = vectorizer.transform(record)
    print(f"  {record['record_id']}: embedding shape = {emb.shape}")

# Use retriever for similarity search
retriever = SpectralRetriever(vectorizer=vectorizer)
retriever.index(records)

# Query with a new record
query = {"record_id": "query", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.015, 0.085, 0.16, 0.11, 0.035, 0.01]}]}
results = retriever.query(query, top_k=3)
print(f"\nNearest neighbors for query:")
for record_id, distance in results:
    print(f"  {record_id}: distance = {distance:.4f}")
