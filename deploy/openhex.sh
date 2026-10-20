#!/usr/bin/env bash
# Our OpenHex bots in a tmux session you can look in on and stop two ways.
#
#   deploy/openhex.sh start [--only NAME]   # start the bots in tmux session "openhex"
#   deploy/openhex.sh attach                # watch the log (detach: Ctrl-B d)
#   deploy/openhex.sh status                # session, pid, each bot's record
#   deploy/openhex.sh drain                 # leave the queues, finish the games in play, exit
#   deploy/openhex.sh kill                  # leave the queues and exit now (forfeits games in play)
#   deploy/openhex.sh leave                 # only leave the queues (after a crash or SIGKILL)
#
# Keys: $OPENHEX_ENV (default ~/.config/catanrl/openhex.env, mode 600) with
# OPENHEX_KEY_1V1, OPENHEX_KEY_1V1_SEARCH, OPENHEX_KEY_4P (deploy/openhex.yaml).
# Device: $OPENHEX_DEVICE (default cuda; cpu where there is no GPU).
# Records: $OPENHEX_STATE (default ~/.local/state/catanrl/openhex): one dir per
# bot with games.jsonl and boards/, and openhex.log, the session's output.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SESSION=openhex
ENV_FILE=${OPENHEX_ENV:-$HOME/.config/catanrl/openhex.env}
STATE=${OPENHEX_STATE:-$HOME/.local/state/catanrl/openhex}
DEVICE=${OPENHEX_DEVICE:-cuda}
CONFIG=deploy/openhex.yaml
MATCH="catanrl openhex $CONFIG"

bot_pid() { pgrep -f "^$ROOT/.venv/bin/python3? .*$MATCH" | head -1; }
load_env() {
    [ -r "$ENV_FILE" ] || { echo "no key file at $ENV_FILE" >&2; exit 1; }
    set -a; . "$ENV_FILE"; set +a
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
}

case ${1:-} in
start)
    shift
    if pid=$(bot_pid) && [ -n "$pid" ]; then echo "already running (pid $pid); attach or stop it first" >&2; exit 1; fi
    tmux kill-session -t "$SESSION" 2>/dev/null || true   # a finished session's pane
    mkdir -p "$STATE"
    load_env
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
        "exec $ROOT/.venv/bin/catanrl openhex $CONFIG --state '$STATE' --device $DEVICE $* 2>&1 | tee -a '$STATE/openhex.log'"
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -n "$(bot_pid)" ] && break; sleep 1; done
    echo "started in tmux session $SESSION (pid $(bot_pid), device $DEVICE); log: $STATE/openhex.log"
    ;;
attach)
    exec tmux attach -t "$SESSION"
    ;;
status)
    if pid=$(bot_pid) && [ -n "$pid" ]; then echo "running: pid $pid, since $(ps -o lstart= -p "$pid")"; else echo "not running"; fi
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session $SESSION is up" || echo "no tmux session"
    for f in "$STATE"/*/games.jsonl; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$(dirname "$f")"): $(wc -l < "$f") games; last:"
        tail -n 1 "$f" | cut -c1-200
    done
    ;;
drain)
    pid=$(bot_pid); [ -n "$pid" ] || { echo "not running" >&2; exit 1; }
    kill -USR1 "$pid" && echo "draining pid $pid: queues left, games in play finish, then it exits (watch: $0 attach)"
    ;;
kill)
    pid=$(bot_pid); [ -n "$pid" ] || { echo "not running" >&2; exit 1; }
    kill -TERM "$pid" && echo "stopping pid $pid now (it leaves the queues on the way out)"
    ;;
leave)
    load_env
    cd "$ROOT" && exec .venv/bin/catanrl openhex "$CONFIG" --leave
    ;;
*)
    sed -n '2,15p' "$0"; exit 2
    ;;
esac
