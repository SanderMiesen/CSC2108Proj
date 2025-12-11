""" visual debugging to check state/action behaviour (also wrt goal conduciveness)"""

def log_entity_positions(env, action, reward, potential_diff=None, r_gc=None, n_episodes=None, step=None):
    """Lightweight text render of entities for debugging."""
    wrapped = getattr(env, "env", env)             # unwrap Nudge env
    level = getattr(wrapped, "level", None)
    if level is None or not hasattr(level, "entities"):
        print("[log_entity_positions] level/entities missing")
        return

    width = int(getattr(wrapped, "width", getattr(level, "width", 50)))
    height = getattr(level, "height", 16)
    rows = 3
    track = [['.'] * width for _ in range(rows)]
    entries = []

    def place_char(row_idx, col_idx, ch):
        if 0 <= row_idx < rows and 0 <= col_idx < width:
            track[row_idx][col_idx] = ch if track[row_idx][col_idx] == '.' else '*'

    for e in level.entities:
        ch = {
            'PLAYER': 'P', 'KEY': 'K', 'DOOR': 'D',
            'GROUND_ENEMY': 'E', 'GROUND_ENEMY2': 'E', 'GROUND_ENEMY3': 'E',
            'BUZZSAW1': 'B', 'BUZZSAW2': 'B',
        }.get(getattr(e._entity_id, "name", "?"), '?')
        col_idx = min(width - 1, max(0, int(round(getattr(e, "x", 0)))))
        row_idx = rows - 1 - min(
            rows - 1,
            max(0, int(getattr(e, "y", 0) / max(1, height) * rows)),
        )
        if getattr(e._entity_id, "name", "") == "PLAYER" and getattr(e, "y", 0) > 2.5:
            row_idx = rows - 2
        place_char(row_idx, col_idx, ch)
        entries.append(f"{getattr(e._entity_id, 'name', '?')}@({getattr(e, 'x', 0):.2f},{getattr(e, 'y', 0):.2f})")

    n_episodes = f"episode={n_episodes} " if n_episodes is not None else ""
    step_str = f"step={step} " if step is not None else ""
    gc_str = f" GC_Progress={r_gc:.2f}" if r_gc is not None else ""
    pd_str = f" potential_diff={potential_diff:.4f}" if potential_diff is not None else ""
    print(f"[Train] {n_episodes} {step_str}reward={reward:.2f}{pd_str}{gc_str} action={action}")
    for row in track:
        print(f"[Train] |{''.join(row)}|")
    print(f"[Train] {' '.join(entries)}")
