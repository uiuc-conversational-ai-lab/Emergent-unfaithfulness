"""
Train faithful RM v2 on GPT-4o-generated preference pairs.

Key fix over v1:
- Training data covers all 10 categories with explicit faithful > unfaithful signal
- No category gets inverted or noise-floor signal (GPT-4o generates clean pairs)
- LoRA fine-tune on allenai/Llama-3.1-Tulu-3-8B-SFT
"""

import json, os, torch, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# All paths are configurable via environment variables. WORK_DIR is a
# writable location for the HF cache, intermediate pairs, and checkpoints.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WORK_DIR  = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))

LOCAL_HF_HOME = os.environ.get("HF_HOME", os.path.join(WORK_DIR, "hf_cache"))
os.environ["HF_HOME"] = LOCAL_HF_HOME
os.makedirs(os.path.join(LOCAL_HF_HOME, "hub"), exist_ok=True)

PAIRS_PATH = os.path.join(WORK_DIR, "faithful_pairs_v2", "rm_train_v2.jsonl")
OUT_DIR    = os.path.join(WORK_DIR, "faithful_rm_v2")
LOG_PATH   = os.path.join(OUT_DIR, "train.log")
RM_BASE    = "allenai/Llama-3.1-Tulu-3-8B-SFT"

EPOCHS      = 2
BATCH_SIZE  = 1
GRAD_ACCUM  = 16
LR          = 1e-4
MAX_LEN     = 512

os.makedirs(OUT_DIR, exist_ok=True)
_log = open(LOG_PATH, "a")

class _Tee:
    def __init__(self, *fs): self.fs = fs
    def write(self, d):
        for f in self.fs: f.write(d); f.flush()
    def flush(self):
        for f in self.fs: f.flush()
    def fileno(self): return self.fs[0].fileno()
    def isatty(self): return False

sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def _encode(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        return self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt")

    def __getitem__(self, idx):
        p = self.pairs[idx]
        chosen   = self._encode(p["chosen"])
        rejected = self._encode(p["rejected"])
        return {
            "chosen_input_ids":      chosen["input_ids"].squeeze(),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(),
            "rejected_input_ids":    rejected["input_ids"].squeeze(),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(),
        }


def load_pairs():
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} RM training pairs")
    return pairs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer from {RM_BASE} ...")
    tok = AutoTokenizer.from_pretrained(RM_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base RM ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE, dtype=torch.float16, device_map="auto", num_labels=1)
    model.config.pad_token_id = tok.pad_token_id

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    pairs = load_pairs()
    dataset = PairDataset(pairs, tok, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    n_steps = (len(loader) * EPOCHS) // GRAD_ACCUM
    print(f"Training: {EPOCHS} epochs, {len(loader)} steps/epoch, {n_steps} optimizer steps (grad_accum={GRAD_ACCUM})")

    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, epoch_acc = 0.0, 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            chosen_ids   = batch["chosen_input_ids"].to(device)
            chosen_mask  = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask= batch["rejected_attention_mask"].to(device)

            chosen_score   = model(chosen_ids,   attention_mask=chosen_mask).logits.squeeze()
            rejected_score = model(rejected_ids, attention_mask=rejected_mask).logits.squeeze()

            # Preference loss: -log(sigmoid(chosen - rejected))
            loss = -torch.nn.functional.logsigmoid(chosen_score - rejected_score).mean()
            acc  = (chosen_score > rejected_score).float().mean().item()

            (loss / GRAD_ACCUM).backward()
            epoch_loss += loss.item()
            epoch_acc  += acc

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_acc  = epoch_acc  / (step + 1)
                    print(f"  step={global_step} loss={avg_loss:.4f} acc={avg_acc:.3f}")

        print(f"Epoch {epoch} complete — step {global_step}")

    # Save
    print(f"Saving faithful RM v2 to {OUT_DIR}")
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
