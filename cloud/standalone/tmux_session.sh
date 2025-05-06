#!/bin/bash
# Start a tmux session for running MARL PPO Suite experiments on Hetzner

SESSION_NAME="marl_experiments"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create new session
    tmux new-session -d -s $SESSION_NAME

    # Set up experiments window
    tmux rename-window -t $SESSION_NAME:0 'experiments'
    tmux send-keys -t $SESSION_NAME:0 'mpo' C-m
    tmux send-keys -t $SESSION_NAME:0 'clear' C-m

    # Create monitoring window with htop
    tmux new-window -t $SESSION_NAME:1 -n 'monitoring'
    tmux send-keys -t $SESSION_NAME:1 'htop' C-m

    # Create GPU monitoring window (if nvidia-smi is available)
    if command -v nvidia-smi &> /dev/null; then
        tmux new-window -t $SESSION_NAME:2 -n 'gpu'
        tmux send-keys -t $SESSION_NAME:2 'watch -n 2 nvidia-smi' C-m
    fi

    # Create logs window to show the latest log content
    tmux new-window -t $SESSION_NAME:3 -n 'logs'
    tmux send-keys -t $SESSION_NAME:3 'mpo' C-m
    tmux send-keys -t $SESSION_NAME:3 'echo "Viewing the most recent log file content..."' C-m
    tmux send-keys -t $SESSION_NAME:3 'latest_log() { find logs -name "*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d" "; }; while true; do LOG=$(latest_log); if [ -n "$LOG" ]; then clear; echo "=== $LOG ==="; tail -n 50 "$LOG"; else echo "No log files found yet"; fi; sleep 2; done' C-m

    # Create disk usage window
    tmux new-window -t $SESSION_NAME:4 -n 'disk'
    tmux send-keys -t $SESSION_NAME:4 'watch -n 60 df -h' C-m

    # Select first window
    tmux select-window -t $SESSION_NAME:0

    # Display help message
    tmux send-keys -t $SESSION_NAME:0 'echo "MARL PPO Suite Experiment Environment"' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo "=================================="' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo "• Run ./run_experiments.sh to start sequential experiments"' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo "• Press Ctrl+B, then number to switch windows"' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo "• Press Ctrl+B, then D to detach (session keeps running)"' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo "• Run tmux attach -t marl_experiments to reconnect later"' C-m
    tmux send-keys -t $SESSION_NAME:0 'echo ""' C-m
fi

# Attach to session
tmux attach-session -t $SESSION_NAME
