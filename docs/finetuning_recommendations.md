# Fine-tuning Recommendations for Improving Specific Error Categories

## Your Training Data Distribution

Based on analysis of `fce_train_bea19.m2`:

| Category       | Total Errors | Blocks  | Pre-FT F0.5 | Post-FT F0.5 | Recommended k |
|----------------|-------------|---------|-------------|--------------|---------------|
| R:VERB:SVA     | 663         | 626     | 0.8072      | 0.7859       | 1 (rarest but works well) |
| R:WO           | 795         | 773     | 0.4464      | 0.4521       | **5** (rare + worst F0.5) |
| U:PREP         | 978         | 931     | 0.6805      | 0.6658       | 2             |
| R:DET          | 1053        | 1021    | 0.5763      | 0.5731       | 3             |
| M:PREP         | 1278        | 1236    | 0.5873      | 0.6148       | 3             |
| R:NOUN:NUM     | 1453        | 1327    | 0.6813      | 0.6735       | 2             |
| R:VERB:TENSE   | 1996        | 1806    | 0.4457      | 0.4915       | **4** (worst recall despite decent data) |
| M:DET          | 2502        | 2193    | 0.7619      | 0.7565       | 1 (common + good) |
| R:PREP         | 2801        | 2544    | 0.7165      | 0.6984       | 1 (most common) |

**Total blocks: 28,350**

**Key Insight**: R:VERB:TENSE has plenty of data (1996 errors) but terrible recall (0.27).
This suggests the model struggles with the *complexity* of tense corrections, not data quantity.
R:WO is both rare (795) AND has terrible recall (0.17) - needs aggressive oversampling.

**Main Problem**: Your uniform `k_target=2` oversampling increased recall but hurt precision 
for categories that were already performing well (SVA, R:PREP).

---

## Recommended Strategy

### Step 1: Category-Specific Oversampling

Use the new `oversample_by_m2_v2.py` with these recommended rates:

```powershell
python utils\finetuning\oversample_by_m2_v2.py `
    --in_m2 datasets\fce_train_bea19.m2 `
    --out_m2 datasets\fce_train_bea19_os.m2 `
    --k_verb_tense 5 `
    --k_word_order 6 `
    --k_m_prep 3 `
    --k_r_det 3 `
    --k_r_prep 1 `
    --k_noun_num 2 `
    --k_u_prep 2 `
    --k_m_det 1 `
    --k_verb_sva 1 `
    --k_other 1 `
    --shuffle `
    --seed 42
```

**Logic:**
- **R:VERB:TENSE (5x)**: Lowest F0.5 (0.4457), lowest recall (0.27)
- **R:WO (6x)**: Second lowest, very low recall (0.17)
- **M:PREP, R:DET (3x)**: Mid-low performance
- **R:NOUN:NUM, U:PREP (2x)**: Moderate improvement needed
- **R:PREP, M:DET, R:VERB:SVA (1x)**: Already good, don't oversample

### Step 2: Adjust Training Parameters

Your current training has some issues:

```powershell
# IMPROVED training command
python train.py `
    --train_set datasets\fce_train_train_bea19_os.txt `
    --dev_set datasets\fce_dev_train_bea19_os.txt `
    --model_dir models\finetuned_v2 `
    --vocab_path data\output_vocabulary `
    --transformer_model roberta `
    --pretrain_folder models `
    --pretrain roberta_1_gectorv2 `
    --tune_bert 1 `
    --lr 5e-6 `
    --batch_size 32 `
    --accumulation_size 2 `
    --n_epoch 5 `
    --patience 2 `
    --max_len 64 `
    --tag_strategy keep_one `
    --skip_correct 0 `
    --skip_complex 0 `
    --special_tokens_fix 1 `
    --label_smoothing 0.1 `
    --predictor_dropout 0.1 `
    --cold_steps_count 1 `
    --cold_lr 1e-4
```

**Key Changes:**
1. **Lower LR (5e-6 vs 1e-5)**: Prevents catastrophic forgetting of good categories
2. **Label smoothing (0.1)**: Reduces overconfidence, improves generalization
3. **Predictor dropout (0.1)**: Prevents overfitting to oversampled data
4. **Accumulation size (2)**: Larger effective batch for stability
5. **Cold steps (1)**: Brief warmup before fine-tuning BERT

### Step 3: Alternative - Mixed Sampling Strategy

Instead of pure oversampling, use a mix:

```powershell
# First, analyze your data distribution
python utils\finetuning\oversample_by_m2_v2.py `
    --in_m2 datasets\fce_train_bea19.m2 `
    --out_m2 datasets\temp.m2 `
    --analyze_only
```

Then adjust multipliers based on the output.

---

## Advanced Technique: Focal Loss (Optional)

If category-specific oversampling isn't enough, consider implementing focal loss
which down-weights easy examples and focuses on hard ones.

Modify `gector/seq2labels_model.py` loss calculation to:

```python
# In forward() method, replace:
loss_labels = sequence_cross_entropy_with_logits(...)

# With focal loss variant (gamma=2 is typical):
def focal_cross_entropy(logits, targets, mask, gamma=2.0):
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                          targets.view(-1), reduction='none')
    pt = torch.exp(-ce)
    focal_loss = ((1 - pt) ** gamma) * ce
    focal_loss = focal_loss.view(mask.size()) * mask.float()
    return focal_loss.sum() / mask.float().sum()
```

---

## Monitoring Tips

During training, watch for:

1. **Validation loss**: Should decrease steadily, not spike
2. **Per-category F0.5 on dev set**: Use errant scorer after each epoch
3. **Precision vs Recall tradeoff**: If precision drops > 5%, reduce oversampling

---

## Quick Reference Commands

```powershell
# 1. Oversample with category-specific rates
python utils\finetuning\oversample_by_m2_v2.py --in_m2 datasets\fce_train_bea19.m2 --out_m2 datasets\fce_train_bea19_os.m2 --k_verb_tense 5 --k_word_order 6 --k_m_prep 3 --k_r_det 3 --shuffle

# 2. Convert to sentence pairs
python utils\finetuning\convert_m2_to_sentence.py

# 3. Preprocess
python utils/preprocess_data.py -s datasets\fce_train_source_bea19_os.txt -t datasets\fce_train_target_bea19_os.txt -o datasets\fce_train_train_bea19_os.txt

# 4. Train with conservative learning rate
python train.py --train_set datasets\fce_train_train_bea19_os.txt --dev_set datasets\fce_dev_train_bea19_os.txt --model_dir models\finetuned_v2 --vocab_path data\output_vocabulary --transformer_model roberta --pretrain_folder models --pretrain roberta_1_gectorv2 --tune_bert 1 --lr 5e-6 --batch_size 32 --n_epoch 5 --patience 2 --label_smoothing 0.1 --predictor_dropout 0.1 --special_tokens_fix 1

# 5. Evaluate
python predict.py --model_path models\finetuned_v2\model.th --vocab_path models\finetuned_v2\vocabulary --input_file test_input.txt --output_file test_output_v2.txt
```
