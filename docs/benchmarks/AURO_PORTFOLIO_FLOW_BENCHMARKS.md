# AURO Portfolio Flow Benchmarks

This suite exercises AURO as a portfolio runtime rather than only as a language-model architecture.

## Covered locally in CI

- AURO-ST-14B cached prefill and single-token decode
- native KV-head storage
- production capability manifest integrity
- browser-conversation archive persistence
- governed Uber/Chrome flow planning that stops before booking
- evidence hashing and repeated sustained execution

## Configured live probes

The workflow supports three external health probes when repository secrets are supplied:

- `NOVA_IOT_BENCHMARK_URL`
- `AURO_BROWSER_BENCHMARK_URL`
- `AURO_CHROME_CDP_BENCHMARK_URL`

A manual workflow run fails closed when the live-gateway job is requested but those endpoints are absent. Pull-request CI skips those probes and records the skips in the evidence report.

## Uber boundary

The benchmark validates the intended Chrome/Uber control sequence:

1. open the Uber surface
2. resolve pickup
3. resolve destination
4. request an estimate
5. select a ride class
6. require explicit human approval
7. stop before booking

It never books, cancels, changes payment, or controls a live account. An authenticated browser gateway and operator approval substrate are required before live interaction can be tested.

## Long execution

Pull requests run a bounded sustained benchmark. Manual runs default to 900 seconds and can be extended through workflow inputs. Each case records iterations, pass/fail/skip counts, median latency, p95 latency, details, and a deterministic evidence hash.

## Hardware boundary

The CPU workflow validates model mechanics and portfolio utilities. It does not measure the full 14B checkpoint or H100 targets. Those require a trained checkpoint, H100 runner, exact serving runtime, and hardware telemetry.
