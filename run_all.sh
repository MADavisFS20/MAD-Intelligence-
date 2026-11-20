#!/usr/bin/env bash
# Kept for old habits: run_all.sh now just calls start.sh (see README.md).
exec "$(dirname "$0")/start.sh" "$@"
