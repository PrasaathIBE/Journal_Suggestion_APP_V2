#!/usr/bin/env bash
set -e

echo "Downloading history.db..."
echo "URL: ${HISTORY_DB_URL}"
echo "PATH: ${SQLITE_PATH:-/tmp/history.db}"

curl -fL "${HISTORY_DB_URL}" -o "${SQLITE_PATH:-/tmp/history.db}"
ls -lh "${SQLITE_PATH:-/tmp/history.db}"

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
