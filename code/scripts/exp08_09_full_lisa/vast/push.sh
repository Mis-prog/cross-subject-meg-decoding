#!/usr/bin/env bash
# С ноутбука: залить репозиторий (код, протоколы, checkout LISA, .env) на сервер vast.ai.
# Использование: bash code/scripts/exp08_09_full_lisa/vast/push.sh <port> <host>     (из "ssh -p PORT root@HOST")
set -euo pipefail
PORT="$1"; HOST="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ssh -p "$PORT" "root@$HOST" 'mkdir -p /workspace/FQW'
rsync -az -e "ssh -p $PORT" \
  --exclude '/.venv/' --exclude '/data/' --exclude '/outputs/' --exclude '/logs/' --exclude '/_to_delete/' \
  --exclude '/docs/' --exclude '/Claude outputs/' --exclude '__pycache__/' --exclude '.mfa_root/' \
  "$ROOT/" "root@$HOST:/workspace/FQW/"
echo "OK: /workspace/FQW на $HOST"
