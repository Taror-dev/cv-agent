#!/usr/bin/env bash
# Installs one local model and checks synthetic DE/RU/EN retrieval.
# Uses an existing loopback server, or stops its own temporary server on exit.
set -euo pipefail
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11434
export PYTHONPATH=src
embedding_server_pid=''
embedding_smoke_dir=$(mktemp -d /tmp/cv-agent-embedding-smoke.XXXXXX)
cleanup() {
  if [ -n "$embedding_server_pid" ]; then
    kill "$embedding_server_pid" 2>/dev/null || true
    wait "$embedding_server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ollama serve >"$embedding_smoke_dir/server.log" 2>&1 &
  embedding_server_pid=$!
  embedding_ready=0
  for embedding_attempt in {1..60}; do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      embedding_ready=1
      break
    fi
    if ! kill -0 "$embedding_server_pid" 2>/dev/null; then
      printf 'Temporary Ollama server exited. Log: %s/server.log\n' "$embedding_smoke_dir" >&2
      exit 2
    fi
    sleep 0.5
  done
  if [ "$embedding_ready" != 1 ]; then
    printf 'Ollama startup timed out. Log: %s/server.log\n' "$embedding_smoke_dir" >&2
    exit 2
  fi
fi
ollama pull qwen3-embedding:0.6b
.venv/bin/python -m cv_agent.embedding_smoke --output "$embedding_smoke_dir/report.json"
printf 'Synthetic embedding report: %s/report.json\n' "$embedding_smoke_dir"
