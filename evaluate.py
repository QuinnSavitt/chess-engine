import os, datetime, torch, chess, chess.pgn, redis
from model import ChessPolicyNet
from move_vocab import MOVE2IDX, IDX2MOVE
from utils import encode_fen_to_tensor, sample_move_from_probs

# Configuration
CHECKPOINT_DIR  = "./checkpoints"
ORIGINAL_MODEL  = "model_original.pt"
LATEST_MODEL    = "model.pt"
HUMAN_GAMES_KEY = "human:games:count"
N_GAMES         = 50
LOG_DIR         = "./logs"

def play_games(model_a, model_b, n_games):
    wins_a = 0
    for _ in range(n_games):
        board = chess.Board()
        turn = 0
        while not board.is_game_over():
            model = model_a if turn % 2 == 0 else model_b
            fen = board.fen()
            x   = encode_fen_to_tensor(fen)
            with torch.no_grad():
                probs = model(x).numpy().ravel()
            mv = sample_move_from_probs(fen, probs)
            board.push_uci(mv)
            turn += 1
        result = board.result()
        if result == "1-0":
            wins_a += 1 if turn % 2 == 1 else 0
        elif result == "0-1":
            wins_a += 1 if turn % 2 == 0 else 0
    return wins_a

def load_model(path):
    model = ChessPolicyNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def main():
    today      = datetime.date.today()
    week_ago   = today - datetime.timedelta(weeks=1)
    two_weeks  = today - datetime.timedelta(weeks=2)

    # 1) Snapshot
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    snap_name = f"model_{today.isoformat()}.pt"
    torch.save(load_model(LATEST_MODEL).state_dict(),
               os.path.join(CHECKPOINT_DIR, snap_name))

    # 2) Load comparators
    orig_path = os.path.join(CHECKPOINT_DIR, ORIGINAL_MODEL)
    original      = load_model(orig_path) if os.path.exists(orig_path) else load_model(LATEST_MODEL)
    lw_path       = os.path.join(CHECKPOINT_DIR, f"model_{week_ago.isoformat()}.pt")
    last_week     = load_model(lw_path) if os.path.exists(lw_path) else None
    tw_path       = os.path.join(CHECKPOINT_DIR, f"model_{two_weeks.isoformat()}.pt")
    two_weeks_ago = load_model(tw_path) if os.path.exists(tw_path) else None

    # 3) Self-play benchmarks
    report_lines = []
    if last_week:
        w = play_games(load_model(LATEST_MODEL), last_week, N_GAMES)
        report_lines.append(f"Vs last week ({week_ago}): {w}/{N_GAMES} wins")
    if two_weeks_ago:
        w = play_games(load_model(LATEST_MODEL), two_weeks_ago, N_GAMES)
        report_lines.append(f"Vs two weeks ago ({two_weeks}): {w}/{N_GAMES} wins")
    w = play_games(load_model(LATEST_MODEL), original, N_GAMES)
    report_lines.append(f"Vs original: {w}/{N_GAMES} wins")

    # 4) Human games count
    rdb       = redis.Redis()
    human_cnt = int(rdb.get(HUMAN_GAMES_KEY) or 0)
    rdb.set(HUMAN_GAMES_KEY, 0)

    # 5) Write to log file
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today.isoformat()}.log")
    with open(log_path, "w") as f:
        f.write("="*40 + "\n")
        f.write(f"Weekly Report: {today.isoformat()}\n")
        f.write(f"Human games processed: {human_cnt}\n\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("="*40 + "\n")

if __name__=="__main__":
    main()
