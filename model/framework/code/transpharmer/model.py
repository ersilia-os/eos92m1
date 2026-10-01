# Adapted without changes from https://github.com/iipharma/transpharmer-repo (MIT licence)
import math
import torch
from torch import nn
from torch.nn import functional as F
from .embed import Embedding


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class KVCache:
    """
    Stores past keys/values for ONE attention layer.
    Tensors are laid out as (B, n_head, T, head_size), so the time axis is dim=2.
    """
    def __init__(self):
        self.key = None
        self.value = None
 
    def update(self, key, value):
        if self.key is None:
            self.key, self.value = key, value
        else:
            # concatenate along the time dimension (dim=2, NOT dim=1 which is heads)
            self.key = torch.cat([self.key, key], dim=2)
            self.value = torch.cat([self.value, value], dim=2)
        return self.key, self.value
 
    def get_cache(self):
        return {"key": self.key, "value": self.value}

    def keep(self, idx):
        """Drop every batch row except ``idx``, which holds row numbers to keep.

        Used when finished sequences are retired mid-generation: the batch axis
        is dim=0, so this keeps all heads/positions/features of the kept rows.
        Advanced indexing copies, so the caller should batch retirements up
        rather than calling this on every decoding step.
        """
        if self.key is not None:
            self.key = self.key[idx]
            self.value = self.value[idx]

    @property
    def seq_len(self):
        return 0 if self.key is None else self.key.size(2)
 
    def reset(self):
        self.key = None
        self.value = None


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.N_EMBD % config.N_HEADS == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.query = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.value = nn.Linear(config.N_EMBD, config.N_EMBD)
        # regularization
        self.attn_drop = nn.Dropout(config.ATTN_PDROP)
        self.resid_drop = nn.Dropout(config.RESID_PDROP)
        # output projection
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
                                     .view(1, 1, config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
        self.n_head = config.N_HEADS

    def forward(self, x, cache=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if cache is not None:
            # prepend the keys/values of every position seen so far in this sample() call
            k, v = cache.update(k, v)
        Tk = k.size(2) # total key length; equals T when there is no cache
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, Tk) -> (B, nh, T, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # the T queries are the *last* T positions of the Tk-long sequence
        att = att.masked_fill(self.mask[:,:,Tk-T:Tk,:Tk] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, Tk) x (B, nh, Tk, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.ln2 = nn.LayerNorm(config.N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.N_EMBD, 4 * config.N_EMBD),
            nn.GELU(),
            nn.Linear(4 * config.N_EMBD, config.N_EMBD),
            nn.Dropout(config.RESID_PDROP))

    def forward(self, x, cache=None):
        y = self.ln1(x)
        y = self.attn(y, cache=cache)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.config = config.MODEL
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.N_LAYERS)])
        self.embed = Embedding(config=self.config, device=self.device)
        self.ln_f = nn.LayerNorm(self.config.N_EMBD)
        self.head = nn.Linear(self.config.N_EMBD, self.config.VOCAB_SIZE, bias=False)
        self.block_size = self.config.MAX_LEN
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def new_caches(self):
        """One KVCache per block, for a single incremental decoding run."""
        return [KVCache() for _ in self.blocks]

    def forward(self, idx, targets=None, prop=None, caches=None):
        # Embedding the whole prefix every step is cheap (one lookup plus the
        # elementwise rotary), and it keeps the rotary positions correct without
        # having to thread an offset through Embedding/TokenEmbedding.
        x = self.embed(token=idx, prop=prop)
        n_props = prop.size(1) if prop is not None else 0
        if caches is not None and caches[0].seq_len:
            # everything but the newest position is already in the cache
            x = x[:, -1:, :]
            n_props = 0
        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if caches is None else caches[i])
        x = self.ln_f(x)
        logits = self.head(x)
        if n_props:
            logits = logits[:, n_props:, :]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample_=False, top_k=None, prop=None,
           use_cache=True, eos_id=None, compact_at=0.75):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time.

    With use_cache=True each step runs only the newest position through the blocks, reusing the
    keys/values of every earlier position, which makes decoding linear rather than quadratic.
    use_cache=False restores the original full-recompute behaviour.

    With eos_id set, rows that emit that token are retired from the batch instead of being
    decoded for the full ``steps``, and their remaining positions are filled with eos_id. This
    is exact because rows never interact: attention is causal and per-row, and no layer
    normalises across the batch, so removing a row cannot change any other row's logits.
    eos_id=None keeps every row alive for all steps (the original behaviour).

    compact_at is the dial on retirement. Removing rows means rebuilding every tensor that has
    a batch axis - x, prop and all 16 cache tensors - which copies rather than views, so it is
    not worth doing for one row at a time. Instead finished rows ride along being computed and
    discarded until they are more than (1 - compact_at) of the batch. Lower = fewer, larger
    copies but more wasted compute in between.
    """
    block_size = model.get_block_size()
    model.eval()
    caches = model.new_caches() if use_cache else None

    B, T0 = x.size()
    # Retired rows never write again, so pre-fill with eos_id and let them keep it. The caller
    # gets the same (B, T0 + steps) block it always did, in the original row order.
    out = x.new_full((B, T0 + steps), eos_id if eos_id is not None else 0)
    out[:, :T0] = x
    # original row number of each row still in the batch; compaction renumbers rows, this
    # remembers where each one belongs in `out`
    alive = torch.arange(B, device=x.device)
    # per-live-row, sticky: a row that has emitted eos_id stays finished
    finished = torch.zeros(B, dtype=torch.bool, device=x.device)

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        # dropping the oldest positions would invalidate the keys/values already cached for them
        assert caches is None or x_cond is x, "context cropping is incompatible with the KV cache"
        logits, _ = model(x_cond, prop = prop, caches = caches)   # for liggpt
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample_:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        if eos_id is not None:
            # A finished row that has not been compacted away yet is still being decoded, so it
            # still draws a token here. Force it back to eos_id, otherwise it could resurrect
            # itself with a non-eos draw and corrupt a sequence that was already complete.
            ix = torch.where(finished.unsqueeze(1), torch.full_like(ix, eos_id), ix)
            finished = finished | (ix.squeeze(1) == eos_id)

        out[alive, T0 + k] = ix.squeeze(1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

        if eos_id is None:
            continue
        live = ~finished
        n_live = int(live.sum())
        if n_live == 0:
            break                                  # every row has completed
        if n_live < compact_at * x.size(0):
            # enough dead weight has built up to be worth the copy: rebuild every batched
            # tensor around the surviving rows
            keep = live.nonzero(as_tuple=True)[0]
            alive, finished, x = alive[keep], finished[keep], x[keep]
            if prop is not None:
                prop = prop[keep]
            for cache in caches or []:
                cache.keep(keep)
    return out