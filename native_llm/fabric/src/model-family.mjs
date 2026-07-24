export function describeModelFamily() {
  return {
    family: 'Auro',
    runtime: 'portable-node-fabric',
    status: 'orchestration-only',
    trained_weights_present: false,
    claims: {
      model_checkpoint: 'not_present',
      tokenizer: 'not_verified',
      official_benchmarks: 'not_verified',
    },
    warning: 'This service is infrastructure and orchestration. It must not be represented as a trained Medina model.',
  };
}
