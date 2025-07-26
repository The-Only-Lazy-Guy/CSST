
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import math
import logging
from torch.utils.checkpoint import checkpoint
import gc

print("Using PyTorch version:", torch.__version__)
logger = logging.getLogger("csst_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
torch._dynamo.config.capture_scalar_outputs = True

DEBUG = False  # Global debug flag
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        return (mem >= threshold).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        threshold = ctx.threshold
        # Improved surrogate gradient with better numerical properties
        steepness = 2.0
        delta = (mem - threshold).clamp(-3.0, 3.0)  # Wider range for stability
        sig = torch.sigmoid(steepness * delta)
        surrogate_grad = steepness * sig * (1 - sig)
        return grad_output * surrogate_grad, None

class VectorizedLIF(nn.Module):
    def __init__(self, dim, threshold=0.15, decay=0.95, target_rate=0.25):
        super().__init__()
        self.dim = dim
        self.decay = decay
        self.linear = nn.Linear(dim, dim, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        
        # Adaptive threshold parameters
        self.register_buffer('base_threshold', torch.tensor(threshold))
        self.register_buffer('threshold', torch.tensor(threshold))
        self.target_rate = target_rate
        self.adaptation_rate = 0.005  # Slower adaptation for stability
        
        # Initialize weights
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        self.linear.weight.data.clamp_(-0.1, 0.1)
        
        # Spike monitoring
        self.register_buffer('spike_activity', None)
        self.register_buffer('last_rate', torch.tensor(0.0))
        self.reset()

    def reset(self):
        self.spike_activity = None
        self.last_rate.fill_(0.0)

    def spike_rate(self):
        return self.last_rate.item()

    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        
        # Stabilize input
        x = torch.tanh(x * 0.5)
        input_current = torch.tanh(self.linear(x))
        
        # Initialize state
        mem = torch.zeros(B, D, device=device)
        spikes = torch.zeros(B, T, D, device=device)
        threshold = self.threshold
        
        # Track average spike rate for adaptation
        rate_accumulator = torch.zeros(1, device=device)
        
        for t in range(T):
            input_t = input_current[:, t]
            
            # Add controlled noise
            noise = torch.randn_like(mem) * 0.005 * threshold
            mem = self.decay * mem + input_t + noise + 1e-6
            
            # Calculate spike
            spike = SpikeFn.apply(mem, threshold)
            
            # Soft reset
            mem = mem * (1 - spike.detach()) - 0.05 * spike.detach()
            spikes[:, t] = spike
            
            # Update rate accumulator
            rate_accumulator += spike.float().mean()
            
            # Update adaptive threshold (slower, every 10 steps)
            if self.training and t % 10 == 0:
                current_rate = rate_accumulator / (t + 1)
                error = current_rate - self.target_rate
                threshold = threshold * (1 + self.adaptation_rate * error)
                threshold = torch.clamp(threshold, 0.1, 0.3)  # Tighter bounds

        # Update class threshold buffer
        if self.training:
            self.threshold = threshold.detach()
        
        # Residual connection
        raw_spike = spikes * input_current * 0.5
        r = torch.sigmoid(self.res_scale)
        out = r * raw_spike + (1 - r) * x
        out = torch.tanh(out * 0.5)  # Stabilize output
        
        # Update spike monitoring
        self.spike_activity = spikes.detach()
        self.last_rate = spikes.mean()

        return out
class EfficientFractalCompress(nn.Module):
    def __init__(self, context_size=512, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.basis = nn.Parameter(torch.randn(context_size, latent_dim))

    def forward(self, x):
        return torch.einsum('bsd,sd->bd', x, self.basis)

class EfficientEntangledAttention(nn.Module):
    def __init__(self, dim, num_heads=4, rank=2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.rank = max(2, dim // 64)
        self.pre_norm = nn.LayerNorm(dim)
        scale = 0.01
        self.entangle_left = nn.Parameter(torch.randn(num_heads, self.head_dim, self.rank) * scale)
        self.entangle_right = nn.Parameter(torch.randn(num_heads, self.rank, self.head_dim) * scale)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(num_heads * self.head_dim, dim)

    def _fused_project(self, x, weight):
        B, T, _ = x.shape
        x_proj = x.view(B, T, self.num_heads, self.head_dim)
        return torch.einsum('bthd,hdr->bthr', x_proj, weight)

    def forward(self, x):
        B, T, _ = x.shape
        x = torch.nan_to_num(x)
        x_norm = self.pre_norm(x)

        q = self._fused_project(self.q_proj(x_norm), self.entangle_left).clamp(-10.0, 10.0)
        k = self._fused_project(self.k_proj(x_norm), self.entangle_left).clamp(-10.0, 10.0)
        v = self._fused_project(self.v_proj(x_norm), self.entangle_left).clamp(-10.0, 10.0)

        attn_scores = torch.einsum('bthr,bths->bhrs', q, k) / math.sqrt(self.rank)
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_out = torch.einsum('bhrs,bths->bthr', attn_probs, v)
        attn_out = torch.einsum('bthr,hrd->bthd', attn_out, self.entangle_right)
        attn_out = attn_out.reshape(B, T, -1)
        out = self.proj(attn_out)

        return torch.nan_to_num(out)

class EfficientHebbian(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight)
        self.norm1 = nn.LayerNorm(in_dim)
        # Hebbian-specific parameters
        self.eta = 0.01
        self.hebb_scale = 1 / math.sqrt(in_dim)
        self.register_buffer('trace', None)
        
    def forward(self, x):
        x = self.norm1(x)
        B, T, D = x.size()
        flat_x = x.view(B * T, D)
        
        # Normal forward pass
        y = F.linear(flat_x, self.weight)
        y_out = y.view(B, T, self.out_dim)
        
        # Hebbian update without affecting autograd graph
        if self.training:
            with torch.no_grad():
                if self.trace is None:
                    self.trace = torch.zeros(self.out_dim, self.in_dim, device=y.device)
                
                # Update trace using exponential moving average
                self.trace = 0.9 * self.trace + self.hebb_scale * torch.mm(y.t(), flat_x)
                
                # Apply normalized update
                update = torch.clamp(self.trace, -1.0, 1.0)
                self.weight.data += self.eta * update
                
                # Normalize columns
                col_norms = self.weight.norm(dim=0, keepdim=True).clamp(min=1e-3)
                self.weight.data.div_(col_norms)

        return y_out

class EfficientUltraBlock(nn.Module):
    def __init__(self, dim, context_size=512, block_id=None, use_checkpoint=False):
        super().__init__()
        self.spike_layer = VectorizedLIF(dim)
        self.hebbian = EfficientHebbian(dim, dim//2)
        self.hebb_proj = nn.Linear(dim//2, dim)
        self.fractal_compress = EfficientFractalCompress(context_size, latent_dim=dim//8)
        self.entangled_attn = EfficientEntangledAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.quantum_scale = nn.Parameter(torch.full((1,), 1e-3))
        self.register_buffer('spike_rate', torch.tensor(0.0), persistent=False)
        self.block_id = block_id
        self.use_checkpoint = use_checkpoint

    def reset(self):
        if hasattr(self.spike_layer, 'reset'):
            self.spike_layer.reset()

    def forward(self, x, global_context):
        spike_output = self.spike_layer(x)
        lif_spike_rate = self.spike_layer.spike_rate()
        self.spike_rate = torch.tensor(lif_spike_rate)

        # Conditional checkpointing
        if self.use_checkpoint:
            hebb_out = checkpoint(self.hebbian, spike_output, use_reentrant=False)
        else:
            hebb_out = self.hebbian(spike_output)
            
        hebb_out = self.hebb_proj(hebb_out)
        attn_in = self.norm(hebb_out)
        
        if self.use_checkpoint:
            attn_out = checkpoint(self.entangled_attn, attn_in, use_reentrant=False)
        else:
            attn_out = self.entangled_attn(attn_in)

        signal_path = attn_out
        if global_context is None:
            compressed_context = self.fractal_compress(x)
        else:
            compressed_context = global_context

        # Add annealed noise
        steps = getattr(self, 'noise_step', 0)
        max_steps = 1000
        annealed_scale = max(0.0, 1 - steps / max_steps)
        self.noise_step = steps + 1
        noise_std = torch.clamp(self.quantum_scale.abs(), min=1e-6, max=0.003) * annealed_scale
        noise = torch.randn_like(signal_path) * noise_std

        gated_out = signal_path + noise
        gated_out = self.norm(gated_out)
        return gated_out, compressed_context

class EfficientCSST_Ultra(nn.Module):
    def __init__(self, vocab_size, dim=512, depth=6, context_size=512, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            EfficientUltraBlock(dim, context_size, block_id=i, use_checkpoint=use_checkpoint)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        self.context_register = nn.Parameter(torch.zeros(1, dim//8))
        self.register_buffer('sleep_cycle', torch.zeros(1), persistent=False)

    def reset(self):
        for m in self.children():
            if hasattr(m, 'reset') and callable(m.reset):
                m.reset()
        for block in self.blocks:
            block.reset()

    def sleep(self, noise_level=0.3):
        with torch.no_grad():
            for param in self.parameters():
                param_mean = param.mean()
                param_std = param.std()
                noise = torch.randn_like(param) * noise_level * param_std
                param.data = param - noise + param_mean
            self.sleep_cycle += 1

    def forward(self, input_ids):
        x = self.embed(input_ids)
        context = self.context_register.expand(x.size(0), -1).clone()
        
        for block in self.blocks:
            block.spike_layer.reset()

        total_spike = 0.0
        for block in self.blocks:
            x, context = block(x, context)
            total_spike = total_spike + block.spike_rate

        avg_spike = total_spike / len(self.blocks)
        x = self.norm(x)
        logits = self.output(x)
        return logits, avg_spike

    def estimate_params(self):
        total = sum(p.numel() for p in self.parameters())
        embed_params = self.embed.weight.numel()
        return total, embed_params, total - embed_params

class MemoryEfficientQAOptimizer(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, tunneling=0.05, fused=True, **kwargs):
        super().__init__(params, lr=lr, fused=fused, **kwargs)
        self.tunneling = tunneling

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    noise = torch.randn_like(p) * self.tunneling
                    p.data.add_(noise)
        return loss

def create_model(vocab_size=50257, **kwargs):
    model = EfficientCSST_Ultra(vocab_size=vocab_size, **kwargs)
    total, embed, net = model.estimate_params()
    logger.info(f"Model created: total={total/1e6:.1f}M params")
    logger.info(f"Embeddings: {embed/1e6:.1f}M | Network: {net/1e6:.1f}M")
    
    # Initialize residual scales to 0.0 for better gradient flow
    for block in model.blocks:
        block.spike_layer.res_scale.data.fill_(0.0)
    
    return model

def train_step(model, batch, optimizer, scaler=None):
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits, spike_rate = model(input_ids)
        targets = input_ids[:, 1:].contiguous().view(-1)
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))

        # Stabilize logits before loss calculation
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        loss = F.cross_entropy(logits, targets, reduction='mean')

    if not torch.isfinite(loss):
        logger.warning(f"⚠️ Skipping NaN/Inf loss: {loss}")
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": float('nan'),
            "spike_rate": spike_rate.item(),
            "sleep_cycle": model.sleep_cycle.item()
        }

    optimizer.zero_grad(set_to_none=True)

    # Add gradient stabilization before clipping
    max_grad_value = 1.0  # Clip individual gradient values
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -max_grad_value, max_grad_value)
            
    # Then proceed with gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Gradient monitoring
    total_grad_norm = 0.0
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Calculate gradient norm before clipping
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        
        # Calculate gradient norm before clipping
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": loss.item(),
        "spike_rate": spike_rate.item(),
        "grad_norm": total_grad_norm,
        "sleep_cycle": model.sleep_cycle.item()
    }

def print_memory_summary():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: Allocated={alloc:.2f}GB, Cached={cached:.2f}GB")
    else:
        logger.info("CUDA not available, memory summary skipped")

def clear_cuda_cache():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")