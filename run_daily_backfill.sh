#!/bin/bash

# Get today's date in YYYY-MM-DD format
TODAY=$(date +%Y-%m-%d)

# Run your Python backfill with start = end = today
python -m y2ai.orchestrator --backfill "$TODAY" "$TODAY"
