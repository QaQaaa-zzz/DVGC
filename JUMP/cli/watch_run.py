#!/usr/bin/env python3
"""JUMP desktop watcher reusing JIT status parsing without simulation imports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from jit_dvgc.error_notifications import check_once, find_completion, find_failure
from jump_planning.protocol import write_json


def notify(title, body):
    subprocess.run(['notify-send', '--app-name=JUMP', '--urgency=critical',
                    title.replace('JIT', 'JUMP'), body], check=True,
                   capture_output=True, text=True, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-run', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=2.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('poll interval must be positive')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.state_dir / 'notification_status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        while True:
            check_once([args.active_run], state, send=notify)
            state.update(application='JUMP', pid=os.getpid(), heartbeat_unix=time.time())
            completed = False
            try:
                completed = find_completion(args.active_run) is not None
                failed = find_failure(args.active_run) is not None
            except (OSError, ValueError):
                failed = False
            state['completion_notified'] = any(row.get('phase') == 'completed' for row in state.get('delivered', {}).values())
            state['error_notified'] = any(row.get('phase') != 'completed' for row in state.get('delivered', {}).values())
            done = ((completed and state['completion_notified']) or (failed and state['error_notified'])) and not state['delivery_errors']
            state['phase'] = 'finished' if done else 'watching'
            write_json(path, state)
            if args.once or done:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
