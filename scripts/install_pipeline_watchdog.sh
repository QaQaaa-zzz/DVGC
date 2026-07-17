#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
install -d "$TARGET"
install -m 0644 "$ROOT/systemd/user/dvgc-pipeline-watchdog.service" "$TARGET/dvgc-pipeline-watchdog.service"
install -m 0644 "$ROOT/systemd/user/dvgc-pipeline-watchdog.timer" "$TARGET/dvgc-pipeline-watchdog.timer"
systemctl --user daemon-reload
systemctl --user enable --now dvgc-pipeline-watchdog.timer
