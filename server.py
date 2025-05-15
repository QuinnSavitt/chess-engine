# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import redis
import io
import chess
import chess.pgn
from model import load_model
from utils import encode_fen_to_tensor, sample_move_from_probs
from move_vocab import MOVE2IDX

app = Flask(__name__)
CORS(app, resources={r"/chess/*": {"origins": ["https://quinnsavitt.com"]}})

rdb       = redis.Redis(host='localhost', port=6379, db=0)
device    = "cpu"
model     = load_model(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# piece values for shaping
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0
}

@app.route("/chess/learn", methods=["POST"])
def learn():
    data   = request.get_json()
    pgn    = data["pgn"]
    result = data["result"]  # "win","loss","draw"
    base   = {"win":1.0, "loss":-1.0, "draw":0.0}[result]

    game = chess.pgn.read_game(io.StringIO(pgn))
    traj = []
    node = game

    # 1) build trajectory with 3-ply lookahead
    while not node.is_end():
        nxt = node.variation(0)
        fen = node.board().fen()
        move = nxt.move
        uci  = move.uci()

        # a) capture reward
        captured = node.board().piece_at(move.to)
        cap_reward = PIECE_VALUES[captured.piece_type] if captured else 0.0

        # b) loss penalty from opponent's previous capture
        prev = node.parent
        if prev is not None:
            prev_move = prev.move
            lost = prev.board().piece_at(prev_move.to)
            loss_penalty = -PIECE_VALUES[lost.piece_type] if lost else 0.0
        else:
            loss_penalty = 0.0

        # c) check bonus
        node.board().push(move)
        check_bonus = 0.05 if node.board().is_check() else 0.0

        # d) 3-ply lookahead material swing
        #    measure material delta after pushing the next 3 moves
        def material_diff(board, depth):
            orig_mat = sum(PIECE_VALUES[p.piece_type] * (1 if p.color == board.turn else -1)
                           for p in board.piece_map().values())
            tmp_node = node
            tmp_board = board.copy()
            for _ in range(depth):
                if tmp_node.is_end(): break
                tmp_node = tmp_node.variation(0)
                tmp_board.push(tmp_node.move)
            new_mat = sum(PIECE_VALUES[p.piece_type] * (1 if p.color == tmp_board.turn else -1)
                          for p in tmp_board.piece_map().values())
            return new_mat - orig_mat

        lookahead3 = material_diff(node.board(), 3) * 0.005

        node.board().pop()

        step_reward = cap_reward + loss_penalty + check_bonus + lookahead3
        traj.append((fen, uci, step_reward))
        node = nxt

    # 2) final outcome bonus
    bonus = len(traj) * 0.005
    total = base + bonus if base <= 0 else base - (bonus / 5)

    # 3) push into Redis
    for fen, uci, step_r in traj:
        rdb.lpush("training:queue", f"{fen}|{uci}|{total + step_r}")

    return jsonify({"status":"queued"})

@app.route("/chess/move", methods=["POST"])
def move():
    data = request.get_json()
    fen  = data.get("fen")
    if fen == "startpos":
        fen = chess.Board().fen()

    x = encode_fen_to_tensor(fen).to(device)
    with torch.no_grad():
        probs = model(x).cpu().numpy().ravel()
    mv = sample_move_from_probs(fen, probs)
    return jsonify({"move": mv})

def background_trainer(batch_size=20):
    import time
    while True:
        items = []
        for _ in range(batch_size):
            it = rdb.rpop("training:queue")
            if not it:
                break
            items.append(it.decode())

        if not items:
            time.sleep(5)
            continue

        # parse batch
        fens, mvs, rews = [], [], []
        for it in items:
            fen, mv, rew = it.split("|")
            fens.append(fen)
            mvs.append(mv)
            rews.append(float(rew))
        rews = torch.tensor(rews, dtype=torch.float32, device=device)
        baseline = rews.mean()

        # compute REINFORCE loss with baseline + entropy
        optimizer.zero_grad()
        loss = 0.0
        for fen, mv, rew in zip(fens, mvs, rews):
            x     = encode_fen_to_tensor(fen).to(device)
            probs = model(x)
            idx   = MOVE2IDX[mv]
            logp  = torch.log(probs[0, idx] + 1e-8)
            loss -= logp * (rew - baseline)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            loss -= 0.01 * entropy

        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), "model.pt")

        time.sleep(5)

if __name__=="__main__":
    import threading
    t = threading.Thread(target=background_trainer, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
