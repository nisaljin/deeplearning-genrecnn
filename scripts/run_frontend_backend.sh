#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  BACKEND_CMD='source .venv/bin/activate && python infer_api.py'
else
  BACKEND_CMD='python infer_api.py'
fi

# Avoid inheriting broken Node runtime flags (e.g. invalid --localstorage-file)
FRONTEND_CMD='env -u NODE_OPTIONS npm run dev'

run_on_macos() {
  osascript - "$PROJECT_ROOT" "$FRONTEND_DIR" "$BACKEND_CMD" "$FRONTEND_CMD" <<'EOF'
on run argv
  set projectRoot to item 1 of argv
  set frontendDir to item 2 of argv
  set backendCmd to item 3 of argv
  set frontendCmd to item 4 of argv

  tell application "Terminal"
    activate
    do script "cd " & quoted form of projectRoot & "; " & backendCmd
    do script "cd " & quoted form of frontendDir & "; " & frontendCmd
  end tell
end run
EOF
}

run_on_linux() {
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "cd \"$PROJECT_ROOT\"; $BACKEND_CMD; exec bash" &
    gnome-terminal -- bash -lc "cd \"$FRONTEND_DIR\"; $FRONTEND_CMD; exec bash" &
    return
  fi

  if command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "cd \"$PROJECT_ROOT\"; $BACKEND_CMD; exec bash" &
    x-terminal-emulator -e bash -lc "cd \"$FRONTEND_DIR\"; $FRONTEND_CMD; exec bash" &
    return
  fi

  echo "No supported terminal launcher found."
  echo "Install gnome-terminal/x-terminal-emulator, or run commands manually:"
  echo "  (cd \"$PROJECT_ROOT\" && $BACKEND_CMD)"
  echo "  (cd \"$FRONTEND_DIR\" && $FRONTEND_CMD)"
  exit 1
}

case "$(uname -s)" in
  Darwin)
    run_on_macos
    ;;
  Linux)
    run_on_linux
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"
    echo "Run these manually:"
    echo "  (cd \"$PROJECT_ROOT\" && $BACKEND_CMD)"
    echo "  (cd \"$FRONTEND_DIR\" && $FRONTEND_CMD)"
    exit 1
    ;;
esac
