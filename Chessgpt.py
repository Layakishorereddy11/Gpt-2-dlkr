from tokenizer import Tokenizer

import chess
import chess.engine
import random
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import numpy as np
import re
from pathlib import Path
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import warnings
import time
warnings.filterwarnings("ignore")
import json
import datetime

    

#----------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 10045
    block_size: int = 256 # maximum context length
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight= self.lm_head.weight

        self.apply(self._init_weights)

    

#----------------------------------------------------

class PGNDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, path: str, n_positions=256):
        self.n_positions = n_positions
        self.tokenizer = tokenizer
        self.games = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.games.append(line)

        

        print("Dataset read, Number of games: ", len(self.games))

    def __pad(self, sample: list):
        while len(sample) < self.n_positions:
            sample.append(self.tokenizer.pad_token_index)
        return sample[:self.n_positions]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        game = self.games[i]
        encoded = self.tokenizer.encode(game, add_bos_token=True)

        if len(encoded) < self.n_positions:
            encoded.append(self.tokenizer.eos_token_index)

        data = self.__pad(encoded)
        return torch.tensor(data)
    
class ChessDataLoaderLite:
    def __init__(self, dataset: PGNDataset, batch_size: int, sequence_length: int, shuffle=True, process_rank = 0, num_processes = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.seed(42)
            random.shuffle(self.indices)
        self.current_idx = self.batch_size * self.process_rank
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
        
    def next_batch(self):
        # Check if we need to reset indices
        if self.current_idx + (self.batch_size * self.num_processes) > len(self.dataset):
            self.current_idx = self.batch_size * self.process_rank
            if self.shuffle:
                random.shuffle(self.indices)
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size * self.num_processes
        
        # Get data from dataset
        batch_data = torch.stack([self.dataset[i] for i in batch_indices])
        
        # Create input and target sequences
        x = batch_data[:, :-1]  # Remove last token for input
        y = batch_data[:, 1:]   # Remove first token for target
        
        # Ensure sequence length is correct
        if x.size(1) > self.sequence_length:
            x = x[:, :self.sequence_length]
            y = y[:, :self.sequence_length]
            
        return x, y




tokenizer = Tokenizer("vocabs/kaggle2_vocab.txt")
dataset = PGNDataset(tokenizer, "dataset/processed_kaggle2.txt", n_positions=129)  # +1 for shifting

# Create dataloader
# loader = ChessDataLoaderLite(dataset, batch_size=4, sequence_length=256)



#-------------------------------------------------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ----------- Checkpoint & Logging Setup Functions -----------
def setup_run(checkpoint_base="checkpoints", logs_base="logs", enable_checkpoints=True):
    """Creates a run-specific checkpoint folder and log file."""
    os.makedirs(logs_base, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = None
    if enable_checkpoints:
        run_dir = os.path.join(checkpoint_base, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        for sub in ["best", "regular", "final"]:
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    log_path = os.path.join(logs_base, f"train_log_{timestamp}.txt")
    return run_dir, log_path

def log_message(log_path, message):
    """Logs a message with timestamp to the log file and prints on console."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(log_path, "a") as fp:
        fp.write(formatted + "\n")

def save_checkpoint(model, optimizer, step, run_dir, device, checkpoint_type="regular"):
    """Save checkpoint in the proper subfolder within the run directory 
       including optimizer state and RNG seeds for exact resumption."""
    subfolder = {"best": "best", "regular": "regular", "final": "final"}.get(checkpoint_type, "regular")
    checkpoint_folder = os.path.join(run_dir, subfolder)
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"model_step_{step}.pt")
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
        "rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available() and "cuda" in device:
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, device):
    """Loads a checkpoint saved with the above function and restores RNG state."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    state_dict = ckpt["model_state_dict"]
    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    # Restore RNG states with proper type conversion
    # rng_state = torch.as_tensor(ckpt["rng_state"], dtype=torch.uint8)
    # torch.set_rng_state(rng_state)
    # np.random.set_state(ckpt["np_rng_state"])
    # if torch.cuda.is_available() and "cuda" in device and "cuda_rng_state" in ckpt:
    #     torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return model, ckpt

# ----------- Validation & Sample Generation Functions -----------
def evaluate_model(model, val_loader, device, num_batches=10):
    """Compute average validation loss across a number of batches."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def generate_sample_response(model, tokenizer, device, prompt="e4 e5", max_length=20, num_return_sequences=1):
    """Generate sample response from the model given a prompt."""
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos_token=True)
    print("tokens: ", tokens)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)
    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            token = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, token), dim=1)
    generated = []
    for i in range(num_return_sequences):
        generated.append(tokenizer.decode(tokens[i].tolist()))
    model.train()
    return generated

# ----------- (Optional) Dataset Splitting Function -----------
def create_data_loaders(dataset, batch_size, sequence_length, val_split=0.1, **loader_kwargs):
    """Splits dataset into training and validation subsets and returns loaders."""
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = ChessDataLoaderLite(train_subset, batch_size=batch_size, sequence_length=sequence_length, **loader_kwargs)
    val_loader = ChessDataLoaderLite(val_subset, batch_size=batch_size, sequence_length=sequence_length, shuffle=False, process_rank=0, num_processes=1)
    return train_loader, val_loader

# ----------- Training Loop Integration -----------
if __name__ == "__main__":
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)


    total_batch_size = 262144 # 2**18, ~0.25M, in number of tokens
    B = 4 # micro batch size
    T = 128 # sequence length

    train_loader, val_loader = create_data_loaders(dataset, batch_size=B, sequence_length=T, val_split=0.1, process_rank=ddp_rank, num_processes=ddp_world_size)

    model=GPT(GPTConfig())
    model.to(device)
    # Another alternative approach
    if device == 'cuda':
        model = torch.compile(model)
    elif device == 'mps':
        # For MPS devices, use 'aot_eager' backend which is more compatible
        model = torch.compile(model, backend='aot_eager')
    else:
        # Skip compilation for CPU
        print("Skipping model compilation for CPU device")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")




    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


    torch.set_float32_matmul_precision('high')

    # create model


    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 250
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # Set up run/log folders. 'enable_checkpoints' can be toggled as needed.
    enable_checkpoints = True  # or set via command-line args
    run_dir, log_path = setup_run(enable_checkpoints=enable_checkpoints)
    log_message(log_path, f"Checkpointing {'enabled' if enable_checkpoints else 'disabled'}.")
    log_message(log_path, f"Using device: {device}")

    checkpoint_interval = 100  # save checkpoint every N steps
    validation_interval = 100  # run validation every N steps
    sample_interval = 100       # generate sample response every N steps

    best_val_loss = float('inf')

    resume_checkpoint = None  # Define resume_checkpoint variable
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        log_message(log_path, f"Resuming training from checkpoint: {resume_checkpoint}")
        model, ckpt = load_checkpoint(resume_checkpoint, device)
        optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.batch_size * train_loader.sequence_length * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        log_message(log_path, f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        # Validation every 100 steps
        if step % validation_interval == 0:
            val_loss = evaluate_model(raw_model, val_loader, device)
            log_message(log_path, f"Validation loss at step {step}: {val_loss:.6f}")
            if val_loss < best_val_loss and enable_checkpoints:
                best_val_loss = val_loss
                ckpt_path = save_checkpoint(raw_model, optimizer, step, run_dir, device, checkpoint_type="best")
                log_message(log_path, f"New best model saved at {ckpt_path}")

        # Sample generation every 50 steps
        if step % sample_interval == 0:
            samples = generate_sample_response(raw_model, tokenizer, device, prompt="e4 e5", max_length=20, num_return_sequences=2)
            log_message(log_path, f"Sample responses at step {step}:")
            for s in samples:
                log_message(log_path, s)

        # Regular checkpoint saving (if desired) every checkpoint_interval steps
        if enable_checkpoints and (step % checkpoint_interval == 0):
            ckpt_path = save_checkpoint(raw_model, optimizer, step, run_dir, device, checkpoint_type="regular")
            log_message(log_path, f"Saved checkpoint at {ckpt_path}")

    # Save final model checkpoint
    if enable_checkpoints:
        final_ckpt = save_checkpoint(raw_model, optimizer, max_steps - 1, run_dir, device, checkpoint_type="final")
        log_message(log_path, f"Final model saved at {final_ckpt}")

    if ddp:
        destroy_process_group()


    #-------------------------------------------------------

    import sys; sys.exit(0)

    # prefix tokens
    model.eval()
    num_return_sequences = 2
    max_length = 5
    tokens = tokenizer.encode("e4 e5", add_bos_token=True)
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits,_ = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(">", decoded)

    # import argparse

    # # Add these arguments near the top of your main training section
    # parser = argparse.ArgumentParser(description="Train ChessGPT")
    # parser.add_argument("--resume_checkpoint", type=str, default=None,
    #                     help="Path to a checkpoint to resume training")
    # args = parser.parse_args()

    # # If resume_checkpoint is provided, load model and optimizer states:
    # if args.resume_checkpoint is not None and os.path.exists(args.resume_checkpoint):
    #     log_message(log_path, f"Resuming training from checkpoint: {args.resume_checkpoint}")
    #     model, ckpt = load_checkpoint(args.resume_checkpoint, device)
    #     # if your checkpoint contains optimizer state use it:
    #     optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #     start_step = ckpt.get("step", 0) + 1