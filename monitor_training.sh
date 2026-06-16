#!/usr/bin/env bash
# Monitor training.log for crashes and checkpoints
set -euo pipefail
cd "$(dirname "$0")"

LOG="${1:-training.log}"
echo "Monitoring $LOG (Ctrl+C to stop)..."

while IFS= read -r line; do
    if echo "$line" | grep -qi "error\|traceback\|exception\|killed\|oom\|cuda"; then
        echo "$(date): CRASH: $line"
        notify-send "Training CRASHED" "$line" 2>/dev/null || true
    elif echo "$line" | grep -q "Saved model"; then
        echo "$(date): CHECKPOINT: $line"
        notify-send "Checkpoint saved" "$line" 2>/dev/null || true
    fi
done < <(tail -f "$LOG")
