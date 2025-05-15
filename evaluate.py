#!/usr/bin/env python3

import os
import datetime
import torch
import chess
import redis

from model import load_model
from move_vocab import MOVE2IDX, IDX2MOVE
from utils import encode_fen_to_tensor, sample_move_from_probs

# Configuration
CHECKPOINT_DIR   = "./checkpoints"
ORIGINAL_MODEL   = "model_original.pt"
LATEST_MODEL     = "model.pt"
HUMAN_GAMES_KEY  = "human:games:count"
N_GAMES          = 50
LOG_DIR          = "./logs"

def play_games(model_a, model_b, n_games):
    wins_a = 0
    for _ in range(n_games):
        board = chess.Board()
        turn  = 0
        # play until terminal
        while not board.is_game_over():
            model = model_a if (turn % 2) == 0 else model_b
            fen   = board.fen()
            x     = encode_fen_to_tensor(fen)
            with torch.no_grad():
                probs = model(x).numpy().ravel()
            mv = sample_move_from_probs(fen, probs)
            board.push_uci(mv)
            turn += 1
        result = board.result()  # "1-0", "0-1", "1/2-1/2"
        # if engine started as White (even turns), "1-0" means engine win
        if result == "1-0":
            wins_a += 1 if (turn % 2) == 1 else 0
        elif result == "0-1":
            wins_a += 1 if (turn % 2) == 0 else 0
    return wins_a

def main():
    today      = datetime.date.today()
    week_ago   = today - datetime.timedelta(weeks=1)
    two_weeks  = today - datetime.timedelta(weeks=2)

    # 1) Snapshot current model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    snap_name = f"model_{today.isoformat()}.pt"
    model_obj = load_model(device="cpu", model_path=LATEST_MODEL)
    torch.save(model_obj.state_dict(),
               os.path.join(CHECKPOINT_DIR, snap_name))

    # --- Prune old checkpoints ---
    files = [f for f in os.listdir(CHECKPOINT_DIR)
             if f.startswith("model_") and f.endswith(".pt")]
    # exclude the original
    files = [f for f in files if f != ORIGINAL_MODEL]
    date_map = {}
    for fn in files:
        ds = fn[len("model_"):-len(".pt")]
        try:
            date_map[fn] = datetime.date.fromisoformat(ds)
        except ValueError:
            pass
    sorted_files = sorted(date_map.items(), key=lambda kv: kv[1])
    keep = {snap_name}
    # last two before current
    prev = [fn for fn, _ in sorted_files if fn != snap_name]
    keep.update(prev[-2:])
    # first-of-month
    month_first = {}
    for fn, dt in sorted_files:
        ym = (dt.year, dt.month)
        if ym not in month_first:
            month_first[ym] = fn
    keep.update(month_first.values())
    # always keep original
    keep.add(ORIGINAL_MODEL)
    # delete others
    for fn in os.listdir(CHECKPOINT_DIR):
        if fn.endswith(".pt") and fn not in keep:
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fn))
            except OSError:
                pass
    # --- End prune block ---

    # 2) Load comparator models
    orig_path = os.path.join(CHECKPOINT_DIR, ORIGINAL_MODEL)
    original      = load_model(device="cpu", model_path=orig_path) \
                    if os.path.exists(orig_path) else model_obj
    lw_path       = os.path.join(CHECKPOINT_DIR, f"model_{week_ago.isoformat()}.pt")
    last_week     = load_model(device="cpu", model_path=lw_path) \
                    if os.path.exists(lw_path) else None
    tw_path       = os.path.join(CHECKPOINT_DIR, f"model_{two_weeks.isoformat()}.pt")
    two_weeks_ago = load_model(device="cpu", model_path=tw_path) \
                    if os.path.exists(tw_path) else None

    # 3) Self-play benchmarks
    report = []
    if last_week:
        w = play_games(model_obj, last_week, N_GAMES)
        report.append(f"Vs last week ({week_ago}): {w}/{N_GAMES} wins")
    if two_weeks_ago:
        w = play_games(model_obj, two_weeks_ago, N_GAMES)
        report.append(f"Vs 2 weeks ago ({two_weeks}): {w}/{N_GAMES} wins")
    w = play_games(model_obj, original, N_GAMES)
    report.append(f"Vs original: {w}/{N_GAMES} wins")

    # 4) Human games count
    rdb       = redis.Redis(host='localhost', port=6379, db=0)
    human_cnt = int(rdb.get(HUMAN_GAMES_KEY) or 0)
    rdb.set(HUMAN_GAMES_KEY, 0)

    # 5) Write report to dated log
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today.isoformat()}.log")
    with open(log_path, "w") as f:
        f.write("="*40 + "\n")
        f.write(f"Weekly Report: {today.isoformat()}\n")
        f.write(f"Human games processed: {human_cnt}\n\n")
        for line in report:
            f.write(line + "\n")
        f.write("="*40 + "\n")

if __name__=="__main__":
    main()
