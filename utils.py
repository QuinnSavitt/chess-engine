import chess
import numpy as np
import torch
import random
from move_vocab import MOVE2IDX

def encode_fen_to_tensor(fen):
    """
    Encode a FEN into an 18×8×8 tensor:
      - 12 planes for pieces
      - 1 plane for side to move
      - 4 planes for castling rights (K/Q for White/Black)
      - 1 plane for en-passant square
    Returns a torch.FloatTensor of shape (1, 18, 8, 8).
    """
    board = chess.Board(fen)

    arr = np.zeros((18, 8, 8), dtype=np.float32)

    # Pieces
    piece_map = {
        'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
        'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11
    }
    for sq, piece in board.piece_map().items():
        idx = piece_map[piece.symbol()]
        r, c = divmod(sq, 8)
        arr[idx, r, c] = 1.0

    # Side to move
    arr[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    arr[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    arr[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    arr[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    arr[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En-passant target
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        arr[17, r, c] = 1.0

    # return (1,18,8,8)
    return torch.from_numpy(arr).unsqueeze(0)


IDX2MOVE = {v: k for k, v in MOVE2IDX.items()}

def sample_move_from_probs(fen: str, probs: np.ndarray) -> str:
    """
    Given a FEN and a flat array of move‐probabilities (shape=(n_moves,)),
    mask out illegal moves, renormalize, and sample one legal UCI string.
    """
    board = chess.Board(fen)
    # list of legal ucis in this position
    legal_ucis = [m.uci() for m in board.legal_moves]

    # map each legal UCI → its index in the probs array,
    # drop any that aren't in your vocabulary
    legal_indices = [MOVE2IDX[uci] for uci in legal_ucis if uci in MOVE2IDX]

    if not legal_indices:
        # unexpected: no intersection between legal moves & vocab
        return random.choice(legal_ucis)

    # extract and renormalize their probabilities
    legal_probs = np.array([probs[i] for i in legal_indices], dtype=np.float32)
    total = legal_probs.sum()
    if total > 0:
        legal_probs /= total
    else:
        # network assigned zero mass to all legal moves → uniform fallback
        legal_probs = np.ones_like(legal_probs) / len(legal_probs)

    # sample one
    choice = np.random.choice(len(legal_indices), p=legal_probs)
    idx = legal_indices[choice]
    return IDX2MOVE[idx]
