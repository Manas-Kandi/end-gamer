#!/usr/bin/env python3
"""Demonstration of the chess environment module."""

import numpy as np
from src.chess_env import Position, MoveEncoder, PositionGenerator


def main():
    """Demonstrate the chess environment module functionality."""
    print("Chess Environment Module Demo")
    print("=" * 40)
    
    # 1. Position Generator Demo
    print("\n1. Position Generator Demo")
    print("-" * 25)
    
    gen = PositionGenerator(curriculum_level=1, seed=42)
    pos = gen.generate_position()
    
    print(f"Generated position (curriculum level 1):")
    print(pos)
    print(f"FEN: {pos.board.fen()}")
    print(f"Valid KP endgame: {gen.is_valid_kp_endgame(pos.board)}")
    
    # 2. Position Tensor Conversion Demo
    print("\n2. Position Tensor Conversion Demo")
    print("-" * 35)
    
    tensor = pos.to_tensor()
    print(f"Tensor shape: {tensor.shape}")
    print(f"Number of pieces: {np.sum(tensor)}")
    
    # Show piece locations
    piece_names = ['King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn']
    for channel in range(12):
        color = 'White' if channel < 6 else 'Black'
        piece = piece_names[channel % 6]
        locations = np.where(tensor[:, :, channel] == 1)
        if len(locations[0]) > 0:
            for row, col in zip(locations[0], locations[1]):
                square_name = chr(ord('a') + col) + str(8 - row)
                print(f"{color} {piece} on {square_name}")
    
    # 3. Move Encoding Demo
    print("\n3. Move Encoding Demo")
    print("-" * 20)
    
    legal_moves = pos.get_legal_moves()
    print(f"Number of legal moves: {len(legal_moves)}")
    
    if legal_moves:
        move = legal_moves[0]
        encoded = MoveEncoder.encode_move(move)
        decoded = MoveEncoder.decode_move(encoded)
        
        print(f"First legal move: {move}")
        print(f"Encoded as: {encoded}")
        print(f"Decoded back: {decoded}")
        print(f"Encoding correct: {move.uci() == decoded.uci()}")
    
    # 4. Move Mask Demo
    print("\n4. Move Mask Demo")
    print("-" * 15)
    
    mask = MoveEncoder.get_move_mask(pos)
    print(f"Move mask shape: {mask.shape}")
    print(f"Number of legal moves (from mask): {int(np.sum(mask))}")
    
    # 5. Policy Vector Demo
    print("\n5. Policy Vector Demo")
    print("-" * 20)
    
    if legal_moves:
        # Create mock visit counts
        visit_counts = [i + 1 for i in range(len(legal_moves))]
        policy = MoveEncoder.moves_to_policy_vector(legal_moves, visit_counts)
        
        print(f"Policy vector shape: {policy.shape}")
        print(f"Policy sum: {np.sum(policy):.6f}")
        
        # Sample a move
        sampled_move = MoveEncoder.policy_vector_to_move(policy, pos, temperature=1.0)
        print(f"Sampled move: {sampled_move}")
    
    # 6. Make Move Demo
    print("\n6. Make Move Demo")
    print("-" * 15)
    
    if legal_moves:
        move = legal_moves[0]
        new_pos = pos.make_move(move)
        
        print(f"Original position:")
        print(pos)
        print(f"\nAfter move {move}:")
        print(new_pos)
        print(f"Original position unchanged: {pos.board.fen() != new_pos.board.fen()}")
    
    # 7. Canonical Form Demo
    print("\n7. Canonical Form Demo")
    print("-" * 22)
    
    canonical = pos.get_canonical_form()
    print(f"Original turn: {'White' if pos.board.turn else 'Black'}")
    print(f"Canonical turn: {'White' if canonical.board.turn else 'Black'}")
    print(f"Same position: {pos.board.fen() == canonical.board.fen()}")
    
    # 8. Curriculum Learning Demo
    print("\n8. Curriculum Learning Demo")
    print("-" * 27)
    
    for level in [0, 1, 2]:
        gen = PositionGenerator(curriculum_level=level, seed=42)
        stats = gen.get_curriculum_stats()
        pos = gen.generate_position()
        
        print(f"Level {level}: {stats['expected_difficulty']}")
        print(f"  Description: {stats['description']}")
        print(f"  Sample position: {pos.board.fen()}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()