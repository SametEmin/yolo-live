#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
OUT="$ROOT/models/yolo11n.onnx"
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
if [[ -f "$OUT" ]]; then
  echo "Model already present: $OUT"
  exit 0
fi
echo "Downloading $URL"
curl -L --fail -o "$OUT" "$URL"
echo "Saved $OUT ($(du -h "$OUT" | cut -f1))"
