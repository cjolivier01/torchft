#!/bin/bash
PIDS="$(ps axu | grep  "${USER}" | grep "train_ddp.py" | awk '{print$2}')"
if [ ! -z "$PIDS" ]; then
  echo "Killing PIDS: $PIDS"
  echo kill -9 $PIDS
fi
