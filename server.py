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

# Redis queue
rdb = redis.Redis(host='localhost', port=6379, db=0)

device    = "cpu"
model     = load_model(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

@app.route("/chess/learn", methods=["POST"])
def learn():
    data   = request.get_json()
    pgn    = data["pgn"]
    result = data["result"]  # "win","loss","draw"
    base   = {"win":1.0, "loss":-1.0, "draw":0.0}[result]

    # Parse moves and add small step rewards
    game = chess.pgn.read_game(io.StringIO(pgn))
    traj = []
    node = game
    while not node.is_end():
        nxt = node.variation(0)
        fen = node.board().fen()
        mv  = nxt.move.uci()

        # step reward: capture or check
        capture_reward = 0.04 if node.board().is_capture(nxt.move) else 0.0
        node.board().push(nxt.move)
        check_reward   = 0.02 if node.board().is_check() else 0.0
        node.board().pop()

        traj.append((fen, mv, capture_reward + check_reward))
        node = nxt

    # final game-level bonus
    bonus = len(traj) * 0.005
    total = base + bonus if base <= 0 else base - (bonus / 5)

    # push each step with total + step
    for fen, mv, step_r in traj:
        rdb.lpush("training:queue", f"{fen}|{mv}|{total + step_r}")

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

        # compute loss with baseline and entropy bonus
        optimizer.zero_grad()
        loss = 0.0
        for fen, mv, rew in zip(fens, mvs, rews):
            x     = encode_fen_to_tensor(fen).to(device)
            probs = model(x)
            idx   = MOVE2IDX[mv]
            logp  = torch.log(probs[0, idx] + 1e-8)
            loss -= logp * (rew - baseline)

            # entropy regularization
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
