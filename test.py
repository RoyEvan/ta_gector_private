def convert_to_utf8(input_path, output_path):
  with open(input_path, "r", encoding="latin-1") as f:
    text = f.read()
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

convert_to_utf8(
  "datasets/wi_locness/ABCN.dev.gold.bea19.src.txt",
  "datasets/wi_locness/ABCN.dev.gold.bea19.src.utf.txt"
)

# convert_to_utf8(
#   "datasets/agentlans/val_preprocessed.txt",
#   "datasets/agentlans/val_preprocessed_utf8.txt"
# )