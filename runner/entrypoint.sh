#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required, e.g. ItsNotAILABS/Auro14B}"
: "${GITHUB_RUNNER_TOKEN:?GITHUB_RUNNER_TOKEN is required}"

RUNNER_NAME="${RUNNER_NAME:-mesie-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,x64,gpu,mesie,spectral-transformer}"
RUNNER_GROUP="${RUNNER_GROUP:-Default}"
EPHEMERAL="${RUNNER_EPHEMERAL:-true}"
REPO_URL="https://github.com/${GITHUB_REPOSITORY}"

cleanup() {
  if [[ -f .runner ]]; then
    ./config.sh remove --unattended --token "${GITHUB_RUNNER_TOKEN}" || true
  fi
}
trap cleanup EXIT INT TERM

python3 -m mesie.training_fabric.discovery \
  --root "${MESIE_WORKSPACE:-/workspace}" \
  --out "${MESIE_RECEIPTS:-/opt/mesie/receipts}/compute-node.json"

args=(
  --unattended
  --url "${REPO_URL}"
  --token "${GITHUB_RUNNER_TOKEN}"
  --name "${RUNNER_NAME}"
  --labels "${RUNNER_LABELS}"
  --runnergroup "${RUNNER_GROUP}"
  --work "${MESIE_WORKSPACE:-/workspace}/_work"
  --replace
)
if [[ "${EPHEMERAL}" == "true" ]]; then
  args+=(--ephemeral)
fi

./config.sh "${args[@]}"
exec ./run.sh
