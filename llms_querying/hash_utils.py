def build_hash_map(route):
    """
    Walk a processed route dict and collect all 'hash' values in DFS order.
    Returns a bidirectional mapping:
        short_to_orig: {"H001": "<original_hash>", ...}
        orig_to_short: {"<original_hash>": "H001", ...}
    Short IDs are fixed-width (H001..H999) so they never partially match each other.
    """
    hashes = []

    def _collect(node):
        if isinstance(node, dict):
            if "hash" in node:
                h = node["hash"]
                if h not in hashes:
                    hashes.append(h)
            for v in node.values():
                _collect(v)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(route)

    short_to_orig = {f"H{i + 1:03d}": h for i, h in enumerate(hashes)}
    orig_to_short = {h: k for k, h in short_to_orig.items()}
    return short_to_orig, orig_to_short


def compress_hashes(text, orig_to_short):
    """Replace original 64-char hashes with short IDs in the input string."""
    for orig, short in orig_to_short.items():
        text = text.replace(orig, short)
    return text


def restore_hashes(text, short_to_orig):
    """Replace short IDs with original hashes in the LLM response string."""
    for short, orig in short_to_orig.items():
        text = text.replace(short, orig)
    return text
