#!/usr/bin/env bash
# MI one-command start:  ./start.sh
#
# Fetches legal open-source Android code (GitHub + Hugging Face), trains your own model from scratch,
# fine-tunes it to answer English requests, and commits progress to your Hugging Face repo as it goes.
# Safe to stop (Ctrl+C) and re-run: it resumes. Extra options are passed to mi_run.py, e.g.
#   ./start.sh --hours 8 --preset small      ./start.sh --no-hf      ./start.sh --help
#
# Hugging Face backup needs a token with WRITE access: https://huggingface.co/settings/tokens
#   export HF_TOKEN=hf_xxx   (or let this script ask once and remember it)
set -u
cd "$(dirname "$0")" || exit 1

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "python3 not found. In Termux: pkg install python"; exit 2; }

# ---- help / status never need setup
for a in "$@"; do case "$a" in -h|--help|--status) exec "$PY" mi_run.py "$@";; esac; done

# ---- dependencies (Termux uses the pkg build of torch; elsewhere pip)
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "Installing PyTorch ..."
  if command -v pkg >/dev/null 2>&1; then pkg install -y python python-numpy python-torch clang
  else "$PY" -m pip install -q torch numpy || "$PY" -m pip install -q --break-system-packages torch numpy
  fi
fi
"$PY" -c "import torch" 2>/dev/null || { echo "PyTorch is still missing; see README.md (Setup)."; exit 2; }
command -v hf >/dev/null 2>&1 || "$PY" -m pip install -q -U huggingface_hub 2>/dev/null \
  || "$PY" -m pip install -q --break-system-packages -U huggingface_hub 2>/dev/null || true

# hf_xet (Hugging Face's optional fast downloader) has no Android build; plain downloads work fine
"$PY" -c "import hf_xet" 2>/dev/null || export HF_HUB_DISABLE_XET=1

# ---- Hugging Face token: environment, saved file, or ask once
TOKEN_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/mi/hf_token"
NO_HF=0; for a in "$@"; do [ "$a" = "--no-hf" ] && NO_HF=1; done
if [ "$NO_HF" = "0" ] && [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN="$(cat "$TOKEN_FILE")"; export HF_TOKEN
  elif ! hf auth whoami >/dev/null 2>&1 && [ -t 0 ]; then
    echo "No Hugging Face login found. Paste a token with WRITE access to back up your model,"
    echo "or just press Enter to train without cloud backup (you can add it later)."
    printf "HF token: "; read -rs HF_TOKEN; echo
    if [ -n "$HF_TOKEN" ]; then
      export HF_TOKEN
      printf "Remember it on this device (file %s, owner-only)? [y/N] " "$TOKEN_FILE"; read -r yn
      if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
        ( umask 077; mkdir -p "$(dirname "$TOKEN_FILE")" && printf "%s" "$HF_TOKEN" > "$TOKEN_FILE" && chmod 600 "$TOKEN_FILE" )
      fi
    else unset HF_TOKEN; fi
  fi
fi
[ -n "${GITHUB_TOKEN:-}" ] || echo "(tip: export GITHUB_TOKEN=... lifts GitHub's 60-requests/hour anonymous limit)"

mkdir -p state logs checkpoints   # the notification's Hold button touches state/HOLD

# ---- keep CPU awake, add foreground notification controls, and run inside tmux
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock
if command -v termux-notification >/dev/null 2>&1; then
  termux-notification --id mi-trainer --ongoing --priority high \
    --title "MI Training Active" \
    --content "Hold: touch '$PWD/state/HOLD' | Resume: rm -f '$PWD/state/HOLD'" \
    --button1 "Hold" --button1-action "touch '$PWD/state/HOLD'" \
    --button2 "Resume" --button2-action "rm -f '$PWD/state/HOLD'"
fi

cleanup() {
  command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock
  command -v termux-notification-remove >/dev/null 2>&1 && termux-notification-remove mi-trainer
}
trap cleanup EXIT

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null 2>&1 && [ -t 0 ]; then
  if tmux has-session -t mi_train 2>/dev/null; then
    echo "Training session 'mi_train' is already running: attaching to it. Options you passed now are ignored."
    echo "(To restart with new options: tmux kill-session -t mi_train, then run this again.)"
  fi
  # tmux re-joins its arguments into one shell string: quote them, and use an absolute path
  # (we already cd'd here, so a relative $0 such as MI/start.sh would no longer resolve).
  SELF="$(pwd)/$(basename "$0")"
  exec tmux new-session -A -s mi_train "$(printf '%q ' bash "$SELF" "$@")"
fi

# ---- run; if the pipeline itself dies unexpectedly, restart it (it resumes from the last checkpoint)
restarts=0; max="${MI_MAX_RESTARTS:-25}"
while true; do
  "$PY" mi_run.py "$@"; rc=$?
  case "$rc" in
    0|2|130) exit "$rc" ;;
  esac
  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$max" ]; then echo "Giving up after $max restarts. See logs/mi_run.log"; exit 1; fi
  echo "mi_run.py exited with $rc; restart $restarts/$max in 30 s (Ctrl+C to stop) ..."
  sleep 30
done
