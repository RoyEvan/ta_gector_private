import argparse
from pathlib import Path

def apply_edits(tokens, edits):
    """
    edits: list of (start, end, replacement_tokens)
    M2 indices are token indices in the original sentence tokens.
    Apply from right to left to keep indices valid.
    """
    for start, end, rep in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        tokens[start:end] = rep
    return tokens

def m2_to_parallel(m2_path: str, out_src: str, out_tgt: str, annotator_id: str = "0"):
    src_lines, tgt_lines = [], []
    s_tokens = None
    edits = []

    def flush():
        nonlocal s_tokens, edits
        if s_tokens is None:
            return
        src = " ".join(s_tokens)
        tgt_tokens = apply_edits(s_tokens.copy(), edits)
        tgt = " ".join(tgt_tokens)
        src_lines.append(src)
        tgt_lines.append(tgt)
        s_tokens = None
        edits = []

    for raw in Path(m2_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue

        if line.startswith("S "):
            flush()
            s_tokens = line[2:].split()
            edits = []

        elif line.startswith("A "):
            # Example:
            # A 5 6|||R:OTHER|||- sized|||REQUIRED|||-NONE-|||0
            parts = line.split("|||")
            span = parts[0].split()  # ["A", start, end]
            start, end = int(span[1]), int(span[2])
            
            if start == -1 and end == -1:
                continue
            
            if parts[1].strip() == "noop":
                continue
            
            replacement = parts[2]  # may be empty
            ann_id = parts[-1]

            if ann_id != annotator_id:
                continue

            rep_tokens = replacement.split() if replacement else []
            edits.append((start, end, rep_tokens))

    flush()
    
    Path(out_src).write_text("\n".join(src_lines) + "\n", encoding="utf-8", errors="ignore")
    Path(out_tgt).write_text("\n".join(tgt_lines) + "\n", encoding="utf-8", errors="ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Category-specific oversampling for GEC")
    ap.add_argument("--m2_path", required=True, help="Input M2 file")
    ap.add_argument("--out_src", required=True, help="Output oversampled M2 file")
    ap.add_argument("--out_tgt", required=True, help="Output oversampled M2 file")
    
    args = ap.parse_args()
    
    # m2_to_parallel("datasets/fce_test_bea19_os.m2", "datasets/fce_test_source_bea19_os.txt", "datasets/fce_test_target_bea19_os.txt", annotator_id="0")
    m2_to_parallel(args.m2_path, args.out_src, args.out_tgt, annotator_id="0")