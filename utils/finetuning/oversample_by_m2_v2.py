"""
Improved oversampling script with category-specific rates and class weighting.
This helps balance rare error categories without over-inflating common ones.
"""
import argparse
import random
from collections import Counter
from typing import List, Set, Dict

def read_m2_blocks(m2_path: str) -> List[List[str]]:
    """Read an M2 file into blocks separated by blank lines."""
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
            err_type = parts[1].strip()
            if err_type != "noop" and err_type != "-NONE-":
                types.append(err_type)
    return types

def normalize_category(err_type: str) -> str:
    """Map error types to their canonical category for statistics."""
    # Determiners/Articles
    if err_type.startswith("M:DET") or err_type == "M:DET":
        return "M:DET"
    if err_type.startswith("R:DET") or err_type == "R:DET":
        return "R:DET"
    if err_type.startswith("U:DET") or err_type == "U:DET":
        return "U:DET"
    
    # Prepositions
    if err_type in ["M:PREP"]:
        return "M:PREP"
    if err_type in ["R:PREP"]:
        return "R:PREP"
    if err_type in ["U:PREP"]:
        return "U:PREP"
    
    # Verbs
    if err_type == "R:VERB:SVA":
        return "R:VERB:SVA"
    if err_type == "R:VERB:TENSE":
        return "R:VERB:TENSE"
    
    # Noun number
    if err_type in ["R:NOUN:NUM", "M:NOUN:NUM"]:
        return "R:NOUN:NUM"
    
    # Word order
    if err_type in ["WO", "R:WO", "M:WO"]:
        return "R:WO"
    
    return err_type

def get_category_for_block(block: List[str], target_categories: Set[str]) -> Set[str]:
    """Get which target categories a block contains."""
    types = extract_err_types(block)
    found = set()
    for t in types:
        norm = normalize_category(t)
        if norm in target_categories:
            found.add(norm)
    return found


def main():
    ap = argparse.ArgumentParser(description="Category-specific oversampling for GEC")
    ap.add_argument("--in_m2", required=True, help="Input M2 file")
    ap.add_argument("--out_m2", required=True, help="Output oversampled M2 file")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle output blocks")
    ap.add_argument("--analyze_only", action="store_true",
                    help="Only print statistics, don't write output")
    
    # Category-specific multipliers (based on your results)
    # Higher multiplier for categories with low recall/F0.5
    ap.add_argument("--k_verb_tense", type=int, default=4,    # Worst F0.5: 0.4457
                    help="Multiplier for R:VERB:TENSE (default: 4)")
    ap.add_argument("--k_word_order", type=int, default=5,    # Second worst: 0.4464
                    help="Multiplier for R:WO (default: 5)")
    ap.add_argument("--k_m_prep", type=int, default=3,        # F0.5: 0.5873
                    help="Multiplier for M:PREP (default: 3)")
    ap.add_argument("--k_r_det", type=int, default=3,         # F0.5: 0.5763
                    help="Multiplier for R:DET (default: 3)")
    ap.add_argument("--k_r_prep", type=int, default=2,        # F0.5: 0.7165
                    help="Multiplier for R:PREP (default: 2)")
    ap.add_argument("--k_noun_num", type=int, default=2,      # F0.5: 0.6813
                    help="Multiplier for R:NOUN:NUM (default: 2)")
    ap.add_argument("--k_u_prep", type=int, default=2,        # F0.5: 0.6805
                    help="Multiplier for U:PREP (default: 2)")
    ap.add_argument("--k_m_det", type=int, default=1,         # Already decent: 0.7619
                    help="Multiplier for M:DET (default: 1)")
    ap.add_argument("--k_verb_sva", type=int, default=1,      # Best: 0.8072
                    help="Multiplier for R:VERB:SVA (default: 1)")
    ap.add_argument("--k_other", type=int, default=1,
                    help="Multiplier for non-target categories (default: 1)")
    
    args = ap.parse_args()
    random.seed(args.seed)

    # Build category -> multiplier mapping
    category_k: Dict[str, int] = {
        "R:VERB:TENSE": args.k_verb_tense,
        "R:WO": args.k_word_order,
        "M:PREP": args.k_m_prep,
        "R:DET": args.k_r_det,
        "R:PREP": args.k_r_prep,
        "R:NOUN:NUM": args.k_noun_num,
        "U:PREP": args.k_u_prep,
        "M:DET": args.k_m_det,
        "R:VERB:SVA": args.k_verb_sva,
    }
    
    target_categories = set(category_k.keys())
    
    blocks = read_m2_blocks(args.in_m2)
    
    # Analyze distribution
    category_counts: Counter = Counter()
    blocks_with_category: Dict[str, int] = Counter()
    
    for block in blocks:
        types = extract_err_types(block)
        seen_in_block = set()
        for t in types:
            norm = normalize_category(t)
            category_counts[norm] += 1
            seen_in_block.add(norm)
        for cat in seen_in_block:
            blocks_with_category[cat] += 1
    
    print("=" * 60)
    print("CATEGORY DISTRIBUTION IN INPUT DATA")
    print("=" * 60)
    print(f"{'Category':<20} {'Total Errors':>12} {'Blocks w/ Cat':>14}")
    print("-" * 60)
    for cat in sorted(target_categories):
        print(f"{cat:<20} {category_counts.get(cat, 0):>12} {blocks_with_category.get(cat, 0):>14}")
    print("-" * 60)
    print(f"{'TOTAL BLOCKS:':<20} {len(blocks):>12}")
    print()
    
    if args.analyze_only:
        return
    
    # Generate output with category-specific oversampling
    out_blocks: List[List[str]] = []
    category_output_counts: Counter = Counter()
    
    for block in blocks:
        cats = get_category_for_block(block, target_categories)
        
        if not cats:
            # No target category - use k_other
            k = args.k_other
        else:
            # Use the maximum multiplier among the categories in this block
            k = max(category_k.get(cat, args.k_other) for cat in cats)
        
        for _ in range(k):
            out_blocks.append(block)
        
        for cat in cats:
            category_output_counts[cat] += k
    
    if args.shuffle:
        random.shuffle(out_blocks)
    
    with open(args.out_m2, "w", encoding="utf-8") as f:
        for block in out_blocks:
            for line in block:
                f.write(line + "\n")
            f.write("\n")
    
    print("=" * 60)
    print("OVERSAMPLING RESULTS")
    print("=" * 60)
    print(f"{'Category':<20} {'Input Blocks':>12} {'Multiplier':>10} {'Output Blocks':>14}")
    print("-" * 60)
    for cat in sorted(target_categories):
        inp = blocks_with_category.get(cat, 0)
        k = category_k.get(cat, args.k_other)
        # Note: actual output depends on overlap with other categories
        print(f"{cat:<20} {inp:>12} {k:>10}x {category_output_counts.get(cat, 0):>14}")
    print("-" * 60)
    print(f"{'Total input blocks:':<20} {len(blocks):>12}")
    print(f"{'Total output blocks:':<20} {len(out_blocks):>12}")
    print(f"Output written to: {args.out_m2}")


if __name__ == "__main__":
    main()
