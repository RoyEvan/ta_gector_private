import argparse
import random
from typing import List, Set

def read_m2_blocks(m2_path: str) -> List[List[str]]:
    """Read an M2 file into blocks separated by blank lines.
    Each block is a list of lines including 'S ...' and 'A ...' lines (without trailing newline).
    """
    blocks: List[List[str]] = []
    cur: List[str] = []
    with open(m2_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "":
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks

def extract_err_types(block: List[str]) -> List[str]:
    """Extract ERRANT error types from A-lines in an M2 block."""
    types = []
    for line in block:
        if not line.startswith("A "):
            continue
        parts = line.split("|||")
        if len(parts) >= 2:
            types.append(parts[1].strip())
    return types

def is_target(err_type: str, target_set: Set[str]) -> bool:
    """Exact or prefix match (use '*' in target_set for prefix)."""
    if err_type in target_set:
        return True
    for t in target_set:
        if t.endswith("*") and err_type.startswith(t[:-1]):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_m2", required=True)
    ap.add_argument("--out_m2", required=True)
    ap.add_argument("--k_target", type=int, default=2)
    ap.add_argument("--k_other", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--shuffle", action="store_true",
                    help="Optional: shuffle output blocks to reduce ordering bias.")
    args = ap.parse_args()

    random.seed(args.seed)

    # Target set for your 6 categories. Adjust after you inspect your M2 labels.
    # From your snippet we confirmed PREP / VERB:SVA / NOUN:NUM exist.
    TARGET_TYPES: Set[str] = {
        "R:VERB:SVA",
        "R:VERB:TENSE",             # likely present in full dataset
        "M:PREP", "R:PREP", "U:PREP",
        "R:NOUN:NUM", "M:NOUN:NUM", # sometimes variants appear
        "WO", "R:WO", "M:WO",
        "M:DET*", "R:DET*", "U:DET*"  # article/determiner variants (prefix)
    }

    blocks = read_m2_blocks(args.in_m2)

    out_blocks: List[List[str]] = []
    n_target = 0

    for block in blocks:
        types = extract_err_types(block)
        has_target = any(is_target(t, TARGET_TYPES) for t in types)
        if has_target:
            n_target += 1
        k = args.k_target if has_target else args.k_other
        for _ in range(k):
            out_blocks.append(block)

    if args.shuffle:
        random.shuffle(out_blocks)

    with open(args.out_m2, "w", encoding="utf-8") as f:
        for block in out_blocks:
            for line in block:
                f.write(line + "\n")
            f.write("\n")  # block separator

    print(f"Input blocks:        {len(blocks)}")
    print(f"Target blocks:       {n_target}")
    print(f"Output blocks:       {len(out_blocks)}")
    print(f"Output written to:   {args.out_m2}")

if __name__ == "__main__":
    main()
