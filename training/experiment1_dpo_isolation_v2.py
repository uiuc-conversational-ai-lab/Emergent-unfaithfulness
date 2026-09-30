"""
=============================================================================
EXPERIMENT 1 v3: DPO-SafetyOnly vs DPO-GeneralOnly (8x H100 version)
=============================================================================

PURPOSE:
--------
Directly answer: "is it the DPO optimizer or the preference data content
that drives AIU?"

Holds the DPO algorithm IDENTICAL across both runs (same LR, beta, epochs,
effective batch size, architecture). Only the DATA differs.

If DPO-SafetyOnly FaithGap >> DPO-GeneralOnly FaithGap:
=> Safety preference DATA, not the DPO optimizer, is the primary driver.

HARDWARE: 8x H100 (80GB each)
------------------------------
You have two execution options. Choose based on your time constraint:

OPTION A - PARALLEL 4+4 (~10 hours total, RECOMMENDED for rebuttal speed):
  Split GPUs: Run A on GPUs 0-3, Run B on GPUs 4-7, simultaneously.
  Need: --gradient_accumulation_steps 32 (4 GPUs x 32 = effective batch 128)
  Need: --cache_reference_model_logprobs (saves memory with only 4 GPUs)
  Both runs finish in ~10 hours simultaneously.

OPTION B - SEQUENTIAL 8+8 (~20 hours total, CLEANEST scientifically):
  Use all 8 GPUs for Run A, then all 8 GPUs for Run B.
  Need: --gradient_accumulation_steps 16 (8 GPUs x 16 = effective batch 128)
  This is IDENTICAL to Tulu 3's original training setup.
  No memory flags needed, most straightforward to justify in paper.

CRITICAL DESIGN: SIZE MATCHING
--------------------------------
Safety data: ~63k pairs
General data (full mixture minus safety): ~290k pairs

MUST sample 63k from general data to match safety size exactly.
Otherwise reviewer argues effect is data quantity, not content.
Same optimizer + same size + different content = clean comparison.

=============================================================================
TRAINING COMMANDS
=============================================================================

--- OPTION A: PARALLEL 4+4 (run both simultaneously in separate terminals) ---

# Terminal 1 - Run A: DPO-SafetyOnly on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --master_port 29500 \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --train_file /path/to/safety_only_63k.jsonl \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --beta 5 \
    --dpo_loss_type dpo_norm \
    --output_dir /path/to/checkpoints/dpo_safety_only \
    --use_flash_attn \
    --cache_reference_model_logprobs \
    --gradient_checkpointing \
    --bf16

# Terminal 2 - Run B: DPO-GeneralOnly on GPUs 4-7 (simultaneously with Run A)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    --master_port 29501 \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --train_file /path/to/general_only_63k.jsonl \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --beta 5 \
    --dpo_loss_type dpo_norm \
    --output_dir /path/to/checkpoints/dpo_general_only \
    --use_flash_attn \
    --cache_reference_model_logprobs \
    --gradient_checkpointing \
    --bf16

NOTE: Different --master_port values (29500 vs 29501) are required to avoid
conflict between the two torchrun processes on the same machine.


--- OPTION B: SEQUENTIAL 8+8 (Tulu 3 exact hyperparameters) ---

# Run A: DPO-SafetyOnly on all 8 GPUs
torchrun --nproc_per_node=8 open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --train_file /path/to/safety_only_63k.jsonl \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --beta 5 \
    --dpo_loss_type dpo_norm \
    --output_dir /path/to/checkpoints/dpo_safety_only \
    --use_flash_attn \
    --bf16

# Run B: DPO-GeneralOnly on all 8 GPUs (after Run A completes)
torchrun --nproc_per_node=8 open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --train_file /path/to/general_only_63k.jsonl \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --beta 5 \
    --dpo_loss_type dpo_norm \
    --output_dir /path/to/checkpoints/dpo_general_only \
    --use_flash_attn \
    --bf16

NOTE: No --cache_reference_model_logprobs needed with 8 GPUs.
      No --gradient_checkpointing needed with 8 GPUs.
      These are IDENTICAL to Tulu 3's original training hyperparameters.

=============================================================================
MEMORY ESTIMATES
=============================================================================

OPTION A (4 GPUs per run):
  Model params sharded:              ~4  GB per GPU
  Optimizer states (AdamW) sharded:  ~16 GB per GPU
  Activations (grad checkpointing):  ~10 GB per GPU
  Cached logprobs buffer:            ~5  GB per GPU
  Total:                             ~35 GB per GPU
  H100 capacity: 80GB => ~45GB headroom. Comfortable.

OPTION B (8 GPUs per run):
  Model params sharded:              ~2  GB per GPU
  Optimizer states sharded:          ~8  GB per GPU
  Activations (no grad checkpoint):  ~15 GB per GPU
  Reference model logprobs:          ~10 GB per GPU
  Total:                             ~35 GB per GPU
  H100 capacity: 80GB => ~45GB headroom. Very comfortable.

=============================================================================
EXPECTED RESULTS
=============================================================================

Model              | Overall Gap | Safety Gap  | Capability  | Subjective
-------------------|-------------|-------------|-------------|------------
SFT Baseline       | +6.3 pp     | large       | medium      | ~0 pp     (existing)
DPO-GeneralOnly    | ~7-9 pp     | small       | small       | ~0 pp     (NEW)
DPO-SafetyOnly     | ~20-25 pp   | large       | small       | ~0 pp     (NEW)
DPO-Full           | +26.1 pp    | large       | medium      | ~0 pp     (existing)

KEY RESULT: DPO-SafetyOnly >> DPO-GeneralOnly
Both use identical DPO setup and identical dataset size.
Only difference: safety content vs. general content.
=> DATA CONTENT is the driver, not the optimizer.

Also check: Does DPO-SafetyOnly FaithGap concentrate in SAFETY categories?
If yes: the mechanism is safety-specific (confirming your paper's claim).
If DPO-GeneralOnly FaithGap concentrates in CAPABILITY categories:
This would confirm that capability-driven unfaithfulness comes from
knowledge-driven deviations (also consistent with your Section 5.3).

=============================================================================
"""

import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths are configurable via environment variables. WORK_DIR is a
# writable location for outputs and checkpoints.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WORK_DIR  = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))

OUTPUT_DIR = os.path.join(WORK_DIR, "experiment1_data")

# Safety classifier cache built by build_safety_classifier_cache.py (this folder)
SAFETY_CLASSIFIER_CACHE = os.environ.get(
    "FAITHCONFLICT_SAFETY_CLASSIFIER_CACHE",
    os.path.join(OUTPUT_DIR, "safety_classifier_cache.jsonl"),
)

# Output paths for the two training datasets
SAFETY_ONLY_OUTPUT  = os.path.join(OUTPUT_DIR, "safety_only_63k.jsonl")
GENERAL_ONLY_OUTPUT = os.path.join(OUTPUT_DIR, "general_only_63k.jsonl")

# Execution option: 'parallel_4_4' or 'sequential_8_8'
EXECUTION_OPTION = "sequential_8_8"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Target size matched to safety data size
# WildGuardMix ~26k + WildJailbreak ~26k + CoCoNot ~11k ≈ 63k
TARGET_SIZE = 63000

# Known safety source identifiers in the preference mixture
# TODO: Verify against actual source field values after running inspect_dataset()
SAFETY_SOURCE_NAMES = [
    'wildguard', 'wildjailbreak', 'coconot',
    'wildguardmix', 'wild_guard', 'wild_jailbreak'
]


# =============================================================================
# STEP 0: INSPECT DATASET - RUN THIS FIRST
# =============================================================================

def inspect_dataset():
    """
    Run this FIRST before anything else.
    Prints field names and sample entries so you can update
    format_preference_item() to match actual data structure.
    
    TODO:
    1. Run this function
    2. Look at printed field names and sample values
    3. Update format_preference_item() accordingly
    4. Look at source distribution to verify SAFETY_SOURCE_NAMES
    """
    print("="*60)
    print("DATASET INSPECTION")
    print("="*60)

    dataset = load_dataset(
        "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        split="train"
    )

    print(f"Total instances: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")
    print()

    print("=== FIRST INSTANCE ===")
    first = dataset[0]
    for key, val in first.items():
        if isinstance(val, str):
            print(f"  {key!r}: {val[:200]!r}")
        elif isinstance(val, list):
            print(f"  {key!r}: list[{len(val)}]")
            if val and isinstance(val[0], dict):
                print(f"    item[0] keys: {list(val[0].keys())}")
                print(f"    item[0] sample: {str(val[0])[:200]}")
        else:
            print(f"  {key!r}: {val!r}")

    print()
    print("=== SOURCE DISTRIBUTION (sample 10k) ===")
    source_counts = defaultdict(int)
    sample_size = min(10000, len(dataset))
    for item in dataset.select(range(sample_size)):
        for field in ['source', 'dataset', 'id']:
            val = item.get(field, '')
            if val and isinstance(val, str):
                key = val.split('/')[0].split('_')[0][:25]
                source_counts[key] += 1
                break
        else:
            source_counts['<no source field>'] += 1

    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {src:<30}: {cnt:>6}")

    print()
    print("TODO: Based on above, update:")
    print("  1. format_preference_item() to use correct field names")
    print("  2. SAFETY_SOURCE_NAMES to match actual source values for safety datasets")


# =============================================================================
# STEP 1: FORMAT PREFERENCE ITEMS
# =============================================================================

def format_preference_item(item, source="unknown"):
    """
    Format a preference pair for open-instruct DPO training.

    TODO: After running inspect_dataset(), update this to match
    the actual field names in the Tulu 3 preference mixture.
    Use the same format as your existing Section 5.3 training data.

    open-instruct dpo_tune_cache.py accepts:

    Format A: {"prompt": str, "chosen": str, "rejected": str}
    Format B: {"messages": [{"role":..., "content":...}], "chosen": str, "rejected": str}
    """
    try:
        # --- FORMAT A: Simple string fields ---
        if all(k in item for k in ['prompt', 'chosen', 'rejected']):
            return {
                "prompt":   item['prompt'],
                "chosen":   item['chosen'],
                "rejected": item['rejected'],
                "source":   source,
            }

        # --- FORMAT B: Messages list ---
        if 'messages' in item and 'chosen' in item and 'rejected' in item:
            return {
                "messages": item['messages'],
                "chosen":   item['chosen'],
                "rejected": item['rejected'],
                "source":   source,
            }

        # --- FORMAT C: Chosen/rejected are message lists ---
        if 'chosen' in item and 'rejected' in item:
            chosen   = item['chosen']
            rejected = item['rejected']
            if isinstance(chosen, list) and len(chosen) >= 1:
                prompt_turns  = chosen[:-1]
                chosen_text   = chosen[-1].get('content', '') if isinstance(chosen[-1], dict) else str(chosen[-1])
                rejected_text = rejected[-1].get('content', '') if isinstance(rejected, list) and rejected and isinstance(rejected[-1], dict) else str(rejected)
                return {
                    "messages": prompt_turns,
                    "chosen":   chosen_text,
                    "rejected": rejected_text,
                    "source":   source,
                }

        # TODO: Add more handlers based on inspect_dataset() output
        return None

    except Exception:
        return None


# =============================================================================
# STEP 2: BUILD SAFETY-ONLY DATASET (~63k pairs)
# =============================================================================

def build_safety_only_dataset():
    """
    Combine the three dedicated Tulu 3 safety preference datasets:
      - WildGuardMix  : ~26k harmful/benign safety pairs
      - WildJailbreak : ~26k adversarial jailbreak pairs
      - CoCoNot       : ~11k contextual noncompliance pairs
    Total: ~63k pairs

    These are the exact datasets that create safety-alignment preference
    signal during DPO, and therefore the candidate driver of AIU.
    """
    print("="*60)
    print("BUILDING SAFETY-ONLY DATASET")
    print("="*60)

    all_safety = []

    # --- WildGuardMix ---
    print("\nLoading WildGuardMix...")
    try:
        wg = load_dataset("allenai/wildguardmixtrain_safety_decontaminated", split="train")
        print(f"  {len(wg)} instances")
        for item in tqdm(wg, desc="  Formatting"):
            f = format_preference_item(item, source="wildguardmix")
            if f: all_safety.append(f)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  TODO: Try 'allenai/wildguardmix' or load from local Section 5.3 copy")

    # --- WildJailbreak ---
    print("\nLoading WildJailbreak...")
    n_before = len(all_safety)
    try:
        wj = load_dataset("allenai/wildjailbreak_safety_decontaminated", split="train")
        print(f"  {len(wj)} instances")
        for item in tqdm(wj, desc="  Formatting"):
            f = format_preference_item(item, source="wildjailbreak")
            if f: all_safety.append(f)
        print(f"  Added {len(all_safety)-n_before} pairs")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  TODO: Try 'allenai/wildjailbreak'")

    # --- CoCoNot (extracted from preference mixture by source field) ---
    print("\nExtracting CoCoNot from preference mixture...")
    n_before = len(all_safety)
    try:
        mix = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train")
        coconot = [
            item for item in mix
            if 'coconot' in str(item.get('source', item.get('dataset', item.get('id', '')))).lower()
        ]
        print(f"  Found {len(coconot)} CoCoNot instances")
        for item in tqdm(coconot, desc="  Formatting"):
            f = format_preference_item(item, source="coconot")
            if f: all_safety.append(f)
        print(f"  Added {len(all_safety)-n_before} pairs")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\nTotal safety pairs collected: {len(all_safety)}")

    # Sample to TARGET_SIZE if over
    if len(all_safety) > TARGET_SIZE:
        all_safety = random.sample(all_safety, TARGET_SIZE)
        print(f"Sampled down to {TARGET_SIZE} for size matching")
    elif len(all_safety) < TARGET_SIZE * 0.8:
        print(f"WARNING: Only {len(all_safety)} safety pairs (expected ~{TARGET_SIZE})")
        print("Check dataset loading above. Proceeding with available data.")

    actual_size = len(all_safety)
    print(f"Final safety dataset size: {actual_size}")
    return all_safety, actual_size


# =============================================================================
# STEP 3: BUILD SIZE-MATCHED GENERAL-ONLY DATASET
# =============================================================================

def build_general_only_dataset(target_size):
    """
    Build a general (non-safety) preference dataset of exactly target_size.

    Process:
    1. Load full Tulu 3 8B preference mixture
    2. Remove safety instances using:
       a. Known safety source names
       b. Your Qwen-3-32B safety classifier labels from Section G
    3. Random sample target_size from remaining general pairs

    Size matching is CRITICAL: without it the comparison is confounded
    by data quantity differences.
    """
    print("="*60)
    print(f"BUILDING GENERAL-ONLY DATASET (target={target_size})")
    print("="*60)

    # Load safety classifier labels if available
    safety_labels = {}
    if SAFETY_CLASSIFIER_CACHE and os.path.exists(SAFETY_CLASSIFIER_CACHE):
        print(f"Loading safety labels from cache...")
        with open(SAFETY_CLASSIFIER_CACHE) as f:
            for line in f:
                item = json.loads(line.strip())
                safety_labels[item['id']] = item.get('safety_label', 0)
        n_safety = sum(1 for v in safety_labels.values() if v == 1)
        print(f"  Loaded {len(safety_labels)} labels, {n_safety} safety-labeled")
    else:
        print(f"WARNING: No safety classifier cache at {SAFETY_CLASSIFIER_CACHE}")
        print("Will use source-name filtering only. Set SAFETY_CLASSIFIER_CACHE for")
        print("more precise filtering (use same labels as Section 5.3).")

    # Load full preference mixture
    print("\nLoading full Tulu 3 preference mixture...")
    full_mix = load_dataset(
        "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        split="train"
    )
    print(f"Full mixture: {len(full_mix)} pairs")

    # Filter out safety instances
    general_items = []
    removed_source = 0
    removed_classifier = 0
    format_failures = 0

    for i, item in enumerate(tqdm(full_mix, desc="Filtering safety instances")):

        item_id     = str(item.get('id', i))
        item_source = str(item.get('source', item.get('dataset', ''))).lower()

        # Filter by known safety source name
        if any(s in item_source for s in SAFETY_SOURCE_NAMES):
            removed_source += 1
            continue

        # Filter by safety classifier
        if safety_labels.get(item_id, 0) == 1:
            removed_classifier += 1
            continue

        formatted = format_preference_item(item, source=item_source or 'general')
        if formatted:
            general_items.append(formatted)
        else:
            format_failures += 1

    print(f"\nRemoved by source name:     {removed_source}")
    print(f"Removed by classifier:      {removed_classifier}")
    print(f"Format failures:            {format_failures}")
    print(f"General instances available: {len(general_items)}")

    # Sample to match safety size exactly
    if len(general_items) >= target_size:
        general_items = random.sample(general_items, target_size)
        print(f"Sampled {target_size} general pairs (size-matched to safety)")
    else:
        print(f"WARNING: Only {len(general_items)} general pairs < target {target_size}")
        print("Using all available. Consider loosening safety filter.")

    return general_items


# =============================================================================
# STEP 4: VERIFY NO OVERLAP AND SAVE
# =============================================================================

def verify_and_save(safety_data, general_data):
    """
    Verify datasets are clean and save them for DPO training.
    Checks: sizes match, no content overlap, required fields present.
    """
    print("="*60)
    print("VERIFICATION AND SAVING")
    print("="*60)

    print(f"Safety dataset:  {len(safety_data)} pairs")
    print(f"General dataset: {len(general_data)} pairs")

    ratio = len(safety_data) / len(general_data) if general_data else 0
    if abs(ratio - 1.0) > 0.05:
        print(f"WARNING: Size ratio {ratio:.3f} differs from 1.0 by >5%")
    else:
        print(f"✓ Sizes matched (ratio={ratio:.4f})")

    # Overlap check via content hash
    def chash(item):
        txt = item.get('chosen', item.get('messages', ''))
        return hash(str(txt)[:200])

    safety_h  = {chash(i) for i in safety_data}
    general_h = {chash(i) for i in general_data}
    overlap   = safety_h & general_h

    if overlap:
        print(f"WARNING: {len(overlap)} overlapping items. Removing from general set.")
        general_data = [i for i in general_data if chash(i) not in overlap]
    else:
        print("✓ No content overlap between datasets")

    # Required fields check
    required_A = {'prompt', 'chosen', 'rejected'}
    required_B = {'messages', 'chosen', 'rejected'}
    for name, data in [("Safety", safety_data), ("General", general_data)]:
        sample = data[:min(10, len(data))]
        has_A = all(required_A.issubset(set(i.keys())) for i in sample)
        has_B = all(required_B.issubset(set(i.keys())) for i in sample)
        if has_A:
            print(f"✓ {name}: Format A (prompt/chosen/rejected)")
        elif has_B:
            print(f"✓ {name}: Format B (messages/chosen/rejected)")
        else:
            print(f"WARNING: {name}: missing required fields. Check format_preference_item().")
            if data:
                print(f"  Sample keys: {list(data[0].keys())}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nSaving to {SAFETY_ONLY_OUTPUT}...")
    with open(SAFETY_ONLY_OUTPUT, 'w') as f:
        for item in safety_data:
            f.write(json.dumps(item) + '\n')
    print(f"✓ {len(safety_data)} pairs saved")

    print(f"Saving to {GENERAL_ONLY_OUTPUT}...")
    with open(GENERAL_ONLY_OUTPUT, 'w') as f:
        for item in general_data:
            f.write(json.dumps(item) + '\n')
    print(f"✓ {len(general_data)} pairs saved")

    return general_data


# =============================================================================
# STEP 5: PRINT EXECUTION PLAN
# =============================================================================

def print_execution_plan(safety_size, general_size):

    print("\n" + "="*70)
    print("EXECUTION PLAN")
    print("="*70)

    if EXECUTION_OPTION == "parallel_4_4":
        print("""
OPTION A SELECTED: PARALLEL 4+4 (~10 hours total)
--------------------------------------------------
Both DPO runs execute SIMULTANEOUSLY on the same 8-GPU machine.
Open two separate terminals and launch both commands at the same time.

  Terminal 1 (GPUs 0-3): DPO-SafetyOnly
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29500 \\
      open_instruct/dpo_tune_cache.py \\
      --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \\
      --train_file {safety_path} \\
      --learning_rate 5e-7 --per_device_train_batch_size 1 \\
      --gradient_accumulation_steps 32 --max_seq_length 2048 \\
      --num_train_epochs 1 --beta 5 --dpo_loss_type dpo_norm \\
      --output_dir {ckpt_safety} \\
      --use_flash_attn --cache_reference_model_logprobs \\
      --gradient_checkpointing --bf16

  Terminal 2 (GPUs 4-7): DPO-GeneralOnly
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29501 \\
      open_instruct/dpo_tune_cache.py \\
      --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \\
      --train_file {general_path} \\
      --learning_rate 5e-7 --per_device_train_batch_size 1 \\
      --gradient_accumulation_steps 32 --max_seq_length 2048 \\
      --num_train_epochs 1 --beta 5 --dpo_loss_type dpo_norm \\
      --output_dir {ckpt_general} \\
      --use_flash_attn --cache_reference_model_logprobs \\
      --gradient_checkpointing --bf16

  Estimated time: ~10 hours for both runs (simultaneously)

  NOTE: Different --master_port values (29500 vs 29501) are required
  to avoid conflict between the two torchrun processes.

  NOTE: --cache_reference_model_logprobs is CRITICAL with 4 GPUs per run.
  It pre-computes reference logprobs, eliminating the reference model from
  GPU memory during training (saves ~16GB per GPU).
""".format(safety_path=SAFETY_ONLY_OUTPUT, general_path=GENERAL_ONLY_OUTPUT,
           ckpt_safety=os.path.join(WORK_DIR, "checkpoints", "dpo_safety_only"),
           ckpt_general=os.path.join(WORK_DIR, "checkpoints", "dpo_general_only")))

    else:  # sequential_8_8
        print("""
OPTION B SELECTED: SEQUENTIAL 8+8 (~20 hours total)
----------------------------------------------------
Use all 8 GPUs for each run. IDENTICAL to Tulu 3 original training setup.
No memory flags needed. Cleanest comparison to cite in paper.

  Run A: DPO-SafetyOnly (8 GPUs, ~10 hours)
  torchrun --nproc_per_node=8 open_instruct/dpo_tune_cache.py \\
      --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \\
      --train_file {safety_path} \\
      --learning_rate 5e-7 --per_device_train_batch_size 1 \\
      --gradient_accumulation_steps 16 --max_seq_length 2048 \\
      --num_train_epochs 1 --beta 5 --dpo_loss_type dpo_norm \\
      --output_dir {ckpt_safety} \\
      --use_flash_attn --bf16

  Run B: DPO-GeneralOnly (8 GPUs, ~10 hours, after Run A)
  torchrun --nproc_per_node=8 open_instruct/dpo_tune_cache.py \\
      --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \\
      --train_file {general_path} \\
      --learning_rate 5e-7 --per_device_train_batch_size 1 \\
      --gradient_accumulation_steps 16 --max_seq_length 2048 \\
      --num_train_epochs 1 --beta 5 --dpo_loss_type dpo_norm \\
      --output_dir {ckpt_general} \\
      --use_flash_attn --bf16

  Estimated time: ~20 hours total

  NOTE: These are IDENTICAL to Tulu 3's original hyperparameters.
  8 GPUs x 1 per_device x 16 grad_accum = 128 effective batch (same as paper).
  No memory flags needed - 8 GPUs has plenty of headroom.
""".format(safety_path=SAFETY_ONLY_OUTPUT, general_path=GENERAL_ONLY_OUTPUT,
           ckpt_safety=os.path.join(WORK_DIR, "checkpoints", "dpo_safety_only"),
           ckpt_general=os.path.join(WORK_DIR, "checkpoints", "dpo_general_only")))

    print("""
AFTER TRAINING - EVALUATION:
-----------------------------
For each checkpoint in [/checkpoints/dpo_safety_only, /checkpoints/dpo_general_only]:

  1. Run FAITHCONFLICT evaluation with your existing pipeline
     (direct prompt, with+without system prompt, Qwen-2.5 32B judge)
     Same as Table 1 in your paper.

  2. Compute FaithGap = FaithRate(confirming) - FaithRate(opposing)
     Report by: Overall / Safety / Capability / Subjective

  3. Fill in this table:
  
  Model              | Overall  | Safety   | Capability | Subjective
  -------------------|----------|----------|------------|------------
  SFT Baseline       | +6.3 pp  | large    | medium     | ~0 pp   (existing)
  DPO-GeneralOnly    | ??? pp   | ???      | ???        | ???     (new)
  DPO-SafetyOnly     | ??? pp   | ???      | ???        | ???     (new)
  DPO-Full           | +26.1 pp | large    | medium     | ~0 pp   (existing)

KEY RESULT TO LOOK FOR:
  DPO-SafetyOnly FaithGap >> DPO-GeneralOnly FaithGap
  DPO-SafetyOnly gap concentrated in SAFETY categories
  DPO-GeneralOnly gap stays near SFT level
  => Safety DATA content (not DPO optimizer) is the primary driver

WHILE TRAINING RUNS: Launch Experiment 2 (RM scoring) in parallel.
  That only needs 1 GPU and finishes in ~2-4 hours.
  python experiment2_rm_scoring.py
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    TODO CHECKLIST - Complete in order:

    BEFORE RUNNING:
    [ ] pip install datasets transformers tqdm
    [ ] huggingface-cli login
    [ ] git clone https://github.com/allenai/open-instruct && pip install -e open-instruct/
    [ ] Set OUTPUT_DIR, SAFETY_CLASSIFIER_CACHE, SAFETY_ONLY_OUTPUT,
        GENERAL_ONLY_OUTPUT, EXECUTION_OPTION at top of file
    [ ] Set EXECUTION_OPTION = "parallel_4_4" (recommended) or "sequential_8_8"

    FIRST RUN (inspection only):
    [ ] Uncomment inspect_dataset() below and comment out rest of main()
    [ ] Run: python experiment1_dpo_isolation_v3.py
    [ ] Look at printed field names and source values
    [ ] Update format_preference_item() and SAFETY_SOURCE_NAMES accordingly
    [ ] Re-comment inspect_dataset() and uncomment rest of main()

    FULL RUN:
    [ ] python experiment1_dpo_isolation_v3.py
    [ ] Verify output files exist and have correct line counts
    [ ] Launch training commands from printed execution plan
    [ ] Monitor loss curves -- both should converge at similar rate
      (if DPO-SafetyOnly converges much faster it may be due to
       simpler distribution; this is fine, just note it)
    [ ] After training: run FAITHCONFLICT evaluation on both checkpoints
    [ ] Fill in results table and include in rebuttal

    REUSE FROM SECTION 5.3:
    [ ] If you have safety classifier labels from Section 5.3 experiments:
        Point SAFETY_CLASSIFIER_CACHE at those labels -- no need to re-run
    [ ] If you have the preference mixture already downloaded:
        datasets will cache automatically, no re-download needed
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # FIRST: Uncomment this, run, check output, then re-comment
    # =========================================================================
    # inspect_dataset()
    # return
    # =========================================================================

    # Step 1: Build safety-only dataset
    safety_data, actual_safety_size = build_safety_only_dataset()

    # Step 2: Build size-matched general-only dataset
    general_data = build_general_only_dataset(target_size=actual_safety_size)

    # Step 3: Verify and save
    general_data = verify_and_save(safety_data, general_data)

    # Step 4: Print execution plan for training
    print_execution_plan(len(safety_data), len(general_data))

    print("\n✓ Data preparation complete.")
    print(f"  Safety-only:  {len(safety_data)} pairs -> {SAFETY_ONLY_OUTPUT}")
    print(f"  General-only: {len(general_data)} pairs -> {GENERAL_ONLY_OUTPUT}")
    print("\nNext step: Follow the training commands in the execution plan above.")
    print("Run Experiment 2 (RM scoring) in parallel while training.")


if __name__ == "__main__":
    main()