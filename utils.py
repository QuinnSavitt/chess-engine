import chess
import numpy as np
import torch
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
