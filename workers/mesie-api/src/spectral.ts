/** Lightweight spectral helpers for the MESIE edge API (TypeScript). */

export type SpectralComponent = {
  name?: string;
  component_id?: string;
  frequency?: number[];
  frequencies?: number[];
  amplitude?: number[];
  amplitudes?: number[];
};

export type SpectralRecord = {
  record_id?: string;
  components?: SpectralComponent[];
  frequency?: number[];
  amplitude?: number[];
};

export type ValidationResult = {
  is_valid: boolean;
  level: number;
  errors: string[];
  warnings: string[];
};

export type MatchResult = {
  composite_score: number;
  metrics: { cosine: number; rmse: number };
  reference_id: string;
  candidate_id: string;
};

function componentArrays(comp: SpectralComponent): { freq: number[]; amp: number[]; name: string } {
  const freq = (comp.frequency ?? comp.frequencies ?? []) as number[];
  const amp = (comp.amplitude ?? comp.amplitudes ?? []) as number[];
  const name = comp.name ?? comp.component_id ?? "component";
  return { freq, amp, name };
}

function primaryComponent(record: SpectralRecord): { freq: number[]; amp: number[]; name: string } {
  if (record.components?.length) {
    return componentArrays(record.components[0]);
  }
  return {
    freq: record.frequency ?? [],
    amp: record.amplitude ?? [],
    name: "component_0",
  };
}

export function validateRecord(payload: unknown): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  let level = 0;

  if (!payload || typeof payload !== "object") {
    return { is_valid: false, level: 0, errors: ["Body must be a JSON object."], warnings: [] };
  }
  level = 1;

  const record = payload as SpectralRecord;
  const { freq, amp, name } = primaryComponent(record);

  if (!freq.length || !amp.length) {
    errors.push(`Component '${name}' missing frequency/amplitude arrays.`);
    return { is_valid: false, level: 2, errors, warnings };
  }
  if (freq.length !== amp.length) {
    errors.push(`Component '${name}' frequency and amplitude length mismatch.`);
    return { is_valid: false, level: 2, errors, warnings };
  }
  level = 2;

  for (let i = 0; i < freq.length; i++) {
    if (!Number.isFinite(freq[i]) || !Number.isFinite(amp[i])) {
      errors.push(`Component '${name}' contains non-finite values.`);
      break;
    }
  }
  for (let i = 1; i < freq.length; i++) {
    if (freq[i] <= freq[i - 1]) {
      warnings.push(`Component '${name}' frequencies are not strictly increasing.`);
      break;
    }
  }
  if (errors.length) {
    return { is_valid: false, level: 2, errors, warnings };
  }
  level = 3;

  return { is_valid: true, level, errors, warnings };
}

function interpolate(target: number[], sourceFreq: number[], sourceAmp: number[]): number[] {
  const out: number[] = [];
  for (const f of target) {
    let j = 0;
    while (j < sourceFreq.length - 1 && sourceFreq[j + 1] < f) j++;
    const f0 = sourceFreq[j];
    const f1 = sourceFreq[Math.min(j + 1, sourceFreq.length - 1)];
    const a0 = sourceAmp[j];
    const a1 = sourceAmp[Math.min(j + 1, sourceAmp.length - 1)];
    if (f1 === f0) {
      out.push(a0);
    } else {
      const t = (f - f0) / (f1 - f0);
      out.push(a0 + t * (a1 - a0));
    }
  }
  return out;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

function rmse(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum / Math.max(a.length, 1));
}

export function matchRecords(reference: SpectralRecord, candidate: SpectralRecord): MatchResult {
  const ref = primaryComponent(reference);
  const cand = primaryComponent(candidate);
  const grid = ref.freq.length >= cand.freq.length ? ref.freq : cand.freq;
  const refAmp = interpolate(grid, ref.freq, ref.amp);
  const candAmp = interpolate(grid, cand.freq, cand.amp);
  const cosine = cosineSimilarity(refAmp, candAmp);
  const err = rmse(refAmp, candAmp);
  const rmseScore = 1 / (1 + err);
  const composite = 0.6 * Math.max(0, cosine) + 0.4 * rmseScore;

  return {
    composite_score: composite,
    metrics: { cosine, rmse: err },
    reference_id: reference.record_id ?? "reference",
    candidate_id: candidate.record_id ?? "candidate",
  };
}