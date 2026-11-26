#!/usr/bin/env bash
# Explicit installation and comparison of two local multilingual models.
# Synthetic data only. Stops only the temporary server started by this script.
set -euo pipefail
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11434
export PYTHONPATH=src
unset CV_AGENT_RUN_ID
comparison_dir=$(mktemp -d /tmp/cv-agent-embedding-comparison.XXXXXX)
comparison_server_pid=''
cleanup() {
  if [ -n "$comparison_server_pid" ]; then
    kill "$comparison_server_pid" 2>/dev/null || true
    wait "$comparison_server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ollama serve >"$comparison_dir/server.log" 2>&1 &
  comparison_server_pid=$!
  comparison_ready=0
  for comparison_attempt in {1..60}; do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      comparison_ready=1
      break
    fi
    if ! kill -0 "$comparison_server_pid" 2>/dev/null; then
      printf 'Temporary Ollama server exited. Log: %s/server.log\n' "$comparison_dir" >&2
      exit 2
    fi
    sleep 0.5
  done
  if [ "$comparison_ready" != 1 ]; then
    printf 'Ollama startup timed out. Log: %s/server.log\n' "$comparison_dir" >&2
    exit 2
  fi
fi
for comparison_model in qwen3-embedding:0.6b bge-m3:latest; do
  printf 'Installing/checking %s (weights stored by Ollama).\n' "$comparison_model"
  if ! ollama pull "$comparison_model" >"$comparison_dir/pull-${comparison_model%%:*}.log" 2>&1; then
    printf 'Model installation failed. Log: %s/pull-%s.log\n' "$comparison_dir" "${comparison_model%%:*}" >&2
    exit 2
  fi
done
PYTHONUNBUFFERED=1 .venv/bin/python -m cv_agent.scoring_evaluation \
  --dataset docs/benchmarks/scoring_tasks_seed.yaml \
  --models docs/benchmarks/local_embedding_models.yaml \
  --k 3 --output "$comparison_dir/report.json"
printf 'Synthetic comparison report: %s/report.json\n' "$comparison_dir"
