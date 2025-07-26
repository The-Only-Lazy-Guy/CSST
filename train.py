
# train.py
import os, json, re, gc, logging, threading, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import datetime

from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing

print(f"[DEBUG] PID={os.getpid()}, Thread={threading.current_thread().name}")
import hashlib
import random

class STEMIterableDataset(IterableDataset):
    def __init__(self, path, tokenizer, max_length=2048, sample=1.0, kfold=None, seed=42):
        """
        Args:
            path (str): Path to JSONL file.
            tokenizer: Tokenizer with `tokenize_academic`.
            max_length (int): Max token length.
            sample (float): Fraction of data to sample (0.0 to 1.0).
            kfold (tuple or None): Tuple of (fold_index, total_folds) for cross-validation.
            seed (int): Random seed for hashing.
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample = sample
        self.kfold = kfold
        self.seed = seed

    def __iter__(self):
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    # Deterministic hash for reproducible folding/sampling
                    uid = hashlib.md5(line.encode('utf-8')).hexdigest()
                    h = int(uid, 16)

                    # Cross-validation fold filtering
                    if self.kfold:
                        idx, total = self.kfold
                        if (h % total) != idx:
                            continue

                    # Random sampling
                    if self.sample < 1.0:
                        rand = (h >> 8) % 1_000_000 / 1_000_000  # Float in [0,1)
                        if rand > self.sample:
                            continue

                    sample = json.loads(line)
                    text = f"Context: {sample.get('context', '')} Question: {sample.get('question', '')} Answer: {sample.get('answer', '')}"
                    toks = self.tokenizer.tokenize_academic(text, max_length=self.max_length)
                    yield {
                        "input_ids": toks['input_ids'].squeeze(0),
                        "attention_mask": toks['attention_mask'].squeeze(0)
                    }

                except Exception:
                    continue

                 # === Tokenizer ===
class AcademicTokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        from transformers import AutoTokenizer
        base = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", use_fast=True)
        super().__init__(
            tokenizer_object=base._tokenizer,
            pad_token=base.pad_token,
            unk_token=base.unk_token,
            eos_token=base.eos_token,
            bos_token=base.bos_token,
            **kwargs
        )
        self.add_special_tokens({"additional_special_tokens": [
            "<math>", "</math>", "<proof>", "</proof>", "<context>", "</context>", "<chem>", "</chem>"
        ]})
        self.add_tokens([
            "∇", "∂", "∫", "∮", "∑", "∏", "∞", "ℵ", "α", "β", "γ", "δ", "ε", "ζ", "η", "θ",
            "ι", "κ", "λ", "μ", "ν", "ξ", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
            "|ψ⟩", "⟨φ|", "⊗", "⨂", "DNA", "RNA", "PCR", "CRISPR", "\\frac", "\\partial", "\\sum", "H₂O", "CO₂"
        ])
        self._tokenizer.post_processor = TemplateProcessing(
            single="<context> $A </context>",
            pair="<context> $A </context> <proof> $B </proof>",
            special_tokens=[
                ("<context>", self.convert_tokens_to_ids("<context>")),
                ("</context>", self.convert_tokens_to_ids("</context>")),
                ("<proof>", self.convert_tokens_to_ids("<proof>")),
                ("</proof>", self.convert_tokens_to_ids("</proof>")),
            ]
        )

    def tokenize_academic(self, text, max_length=8192):
        text = re.sub(r'(\$[^\$]+\$)', r'<math> \1 </math>', text)
        text = re.sub(r'([A-Z][a-z]?\d*\w*)', r'<chem> \1 </chem>', text)
        return self(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    
class SpikeMonitor:
    def __init__(self, threshold=0.1, rate=0.95): 
        self.threshold, self.rate, self.count = threshold, rate, 0
        
    def check(self, model):
        r = model.blocks[0].spike_layer.spike_rate()
        if r < self.threshold:
            self.count += 1
            if self.count > 3:
                for b in model.blocks:
                    b.spike_layer.threshold *= self.rate
                self.count = 0
                return True
        else: 
            self.count = max(0, self.count - 1)
        return False

def main():
    from model import create_model, MemoryEfficientQAOptimizer, train_step, clear_cuda_cache
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import time
    import threading
    from datetime import datetime
    from tqdm import tqdm
    import torch.nn.functional as F

    logging.info(f"MAIN PID={os.getpid()} Thread={threading.current_thread().name}")
    plt.clf()  # Clear current figure

    def adjust_learning_rate(optimizer, step, initial_lr=2e-6, warmup_steps=100):
        """Implements linear learning rate warmup"""
        lr = min(initial_lr * (step + 1) / warmup_steps, initial_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    def plot_metrics(epoch=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.figure(figsize=(16, 9))
        plt.plot(losses, label='Train Loss')
        plt.plot(global_val_losses, label='Val Loss')
        plt.title('Loss Curve'); plt.xlabel('Step'); plt.ylabel('Loss'); plt.legend()
        plt.savefig(os.path.join(plot_dir, f"loss_plot_ep{epoch}_{timestamp}.png")); plt.close()

        plt.figure(figsize=(16, 9))
        plt.plot(gpu_memories, label='Peak GPU Memory (MB)', color='red')
        plt.title('GPU Memory Usage'); plt.xlabel('Steps'); plt.ylabel('Memory (MB)'); plt.legend()
        plt.savefig(os.path.join(plot_dir, f"gpu_memory_plot_{timestamp}.png")); plt.close()

        plt.figure(figsize=(16, 9))
        plt.plot(Spike_Rates, label='Spike Rate', color='green')
        plt.title('Spike Rate'); plt.xlabel('Steps') ; plt.legend()
        plt.savefig(os.path.join(plot_dir, f"spike_rate_{timestamp}.png")); plt.close()

        logging.info(f"Plots saved to {plot_dir}")


    # === Training ===
    def train(model, loader, optimizer, scaler, monitor, sleep_interval=500, accumulate_steps=2, max_steps_per_epoch=5000):
        model.train()
        total_loss = 0.0
        torch.cuda.reset_peak_memory_stats()
        tokenizer = tok
        pad_id = tokenizer.pad_token_id

        for step, batch in enumerate(loader):
            if step > max_steps_per_epoch:
                logging.warning(f"Step cap {max_steps_per_epoch} reached, skipping remainder of epoch.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            if batch['attention_mask'].sum() == 0:
                logging.warning(f"Empty attention mask at step {step}")
                continue
                
            # Use the model's train_step function
            metrics = train_step(model, batch, optimizer, scaler)
            
            if torch.isnan(torch.tensor(metrics['loss'])):
                logging.warning(f"Skipping step {step} due to NaN loss")
                continue
                
            total_loss += metrics['loss']
            avg_loss = total_loss / (step + 1)
            
            losses.append(metrics['loss'])
            gpu_memories.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
            Spike_Rates.append(metrics['spike_rate'])
            
            # Log gradient norm
            logging.info(f"[step {step}] Loss: {metrics['loss']:.4f} | "
                         f"Spike: {metrics['spike_rate']:.4f} | "
                         f"Grad Norm: {metrics['grad_norm']:.4f}")
            if step % 5 == 0:
                logging.info(
                    f"[step {step}] Threshold: {model.blocks[0].spike_layer.threshold.item():.4f} | "
                )
            # Noise injection if under-spiking
            if metrics['spike_rate'] < SPIKE * 0.8:
                mask = batch['input_ids'] != pad_id
                noise = torch.randint(0, 4, batch['input_ids'].shape, device=batch['input_ids'].device)
                batch['input_ids'] = torch.where(mask, 
                                                torch.clamp(batch['input_ids'] + noise, 
                                                            max=tokenizer.vocab_size - 1), 
                                                batch['input_ids'])

            # Adaptive threshold decay
            if metrics['spike_rate'] < SPIKE * 0.7:
                for blk in model.blocks:
                    blk.spike_layer.threshold *= 0.9

            yield step, metrics['loss'], metrics['spike_rate']

            # Validation and checkpointing
            if step % 5 == 0 and step > 0:
                validate_and_checkpoint(model, step, optimizer)
                plot_metrics()

            

        return total_loss / len(loader)

    def validate_and_checkpoint(model, step, optimizer):
        logging.info(f"[step {step}] Running Validation...")
        model.eval()
        val_losses = []
        
        with autocast(device_type="cuda", enabled=True):
            for i, val_batch in enumerate(val_loader):
                if i >= VAL_BATCHES:
                    break
                    
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                logits, _ = model(val_batch['input_ids'])
                targets = val_batch['input_ids'][:, 1:].contiguous().view(-1)
                logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                loss = F.cross_entropy(logits, targets, reduction='mean')
                loss = loss * 0.9 + 0.1 * loss.detach()  # Smooth loss fluctuations

                
                val_losses.append(loss.item())
                
        avg_val_loss = np.mean(val_losses)
        global_val_losses.append(avg_val_loss)
        logging.info(f"[Validation] Step {step} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'val_loss': avg_val_loss,
            'spike_rates': [b.spike_layer.spike_rate() for b in model.blocks]
        }
        ckpt_path = f"models/csst_step{step}.pth"
        torch.save(ckpt, ckpt_path)
        logging.info(f"Checkpoint saved to {ckpt_path}")
        
        model.train()
        return avg_val_loss

    # === Watchdog ===
    def watchdog():
        while True:
            logging.info(f"[{datetime.now()}] Training still active - steps: {len(losses)}")
            time.sleep(300)
    
    # Configurations
    BATCH, ACC, EPOCHS = 2, 4, 3
    LR, MAXL, SLEEP, SPIKE = 2e-4, 1024, 500, 0.3 # LR is changing, the given is initial
    VAL_BATCHES = 10  # Number of validation batches
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler(device='cuda', enabled=True, growth_interval=100)

    is_dist = (torch.cuda.device_count() > 1)
    
    if is_dist: 
        dist.init_process_group('nccl')
        logging.info("Distributed training initialized")

    # Start watchdog
    threading.Thread(target=watchdog, daemon=True).start()

    # Initialize tokenizer and datasets
    tok = AcademicTokenizer()  # Assuming this is defined elsewhere
    trds = STEMIterableDataset("data/raw_data/train.jsonl", tok, MAXL)
    valds = STEMIterableDataset("data/raw_data/val.jsonl", tok, MAXL)
    
    trdl = DataLoader(trds, batch_size=BATCH, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valds, batch_size=BATCH, num_workers=2, pin_memory=True)

    # Create model
    model = create_model(vocab_size=len(tok), dim=512, depth=6, context_size=MAXL).to(device)
    
    if is_dist:
        model = FSDP(model, mixed_precision=True)


    opt = MemoryEfficientQAOptimizer(model.parameters(), lr=LR, tunneling=0.01, fused=False)
    mon = SpikeMonitor(threshold=SPIKE)

    # === Training Loop ===
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\nEpoch {epoch}/{EPOCHS}")
        
        train_gen = train(model, trdl, opt, scaler, mon, 
                         sleep_interval=SLEEP, 
                         accumulate_steps=ACC)
        
        for step, loss, spike_rate in train_gen:
            # Training happens inside the generator
            pass
            
        # End of epoch validation
        validate_and_checkpoint(model, f"epoch_{epoch}", opt)
        
    # Final save
    torch.save(model.state_dict(), "models/csst_final.pth")
    tok.save_pretrained("models/tokenizer")
    logging.info("Training completed successfully")
    
    if is_dist: 
        dist.destroy_process_group()

    

# === Main ===
if __name__ == '__main__':
    # === Env Setup ===
    os.environ["TRITON_CACHE_DIR"] = "E:/short_triton_cache"
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # === Logging ===
    os.makedirs("training_logs", exist_ok=True)
    log_file = f"training_logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("Training session started.")

    # === Plotting ===
    plot_dir = "training_plots"
    os.makedirs(plot_dir, exist_ok=True)
    losses, global_val_losses, gpu_memories = [], [], []
    Spike_Rates = []
    
    main()

    