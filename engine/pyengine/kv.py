"""B3.1: KV + recurrent state. Three stores, parity semantics with vLLM's
per-layer-type specs (BASELINE_NOTES.md; vllm inkling nvidia/attention.py:
200-211 returns SlidingWindowSpec/FullAttentionSpec per layer type, sconv a
window-4 spec): paged global KV (PAGE_SIZE 16, 8 KV heads x hd128, 11
layers), ring-512 SWA KV (16 KV heads, 55 layers), and sconv rings (window-4
raw-input tails, 4 sites per layer: k, v, attn-out, mlp-out). Caches hold
POST-q/k-norm K and POST-sconv V — per-token values that never change once
written (norm/sconv are causal), so decode never re-normalizes history.
Correctness oracle (t_b3 kv): decode(N+1) == recompute-from-scratch across
the 512 edge. B2.7's boundary fact: a token influences SWA output for
window + sconv_k - 1 = 515 positions — the last sconv_k-1 = 3 RAW pre-conv
inputs live in the SconvRing, separate state from the 512 post-conv KV
entries. Single-sequence scope here; the B3.3 scheduler owns multi-seq."""
import torch

PAGE_SIZE = 16  # tokens per global-KV page (PROGRESS B3.1 "paged global (page 16)")


class SconvRing:
    """Rolling tail of the last (kernel-1) RAW inputs of one sconv site.
    Zero-initialized: absent history == F.conv1d's implicit left zero-pad,
    so a fresh ring is exactly "before t=0" (model.sconv_prefill semantics,
    modeling_inkling.py:461-481)."""

    def __init__(self, channels, kernel, device, dtype=torch.bfloat16,
                 backing=None):
        #1. tail[i] = raw input at backward distance kernel-1-i (tail[-1]
        #   is the most recent token); zeros = implicit conv padding.
        #   `backing` = a [kernel-1, channels] view of one SconvPool row
        #   (exp-0008): the conv window READS every tail row, so unlike
        #   the KV pools a prior occupant's stale values cannot be masked
        #   away — the row is erased here so a fresh ring is zeros either
        #   way; step/seed then write THROUGH the view, keeping the pool
        #   and the per-seq code paths on one storage
        self._backed = backing is not None
        if self._backed:
            self.tail = backing
            self.tail.zero_()
        else:
            self.tail = torch.zeros(kernel - 1, channels, device=device,
                                    dtype=dtype)

    def seed(self, x):
        """Load the tail from a prefill's raw input stream x [T, C]; keeps
        the leading zeros when T < kernel-1 (short prompts)."""
        #1. the last min(T, kernel-1) rows are the live history
        n = min(x.shape[0], self.tail.shape[0])
        if n:
            self.tail[-n:] = x[-n:]

    def step(self, x):
        """One decode token: x [1, C] raw input -> the full conv window
        [kernel, C] (oldest first, x last), rolling the tail forward."""
        #1. window = kept tail + the new token; new tail drops the oldest
        #   (in place when pool-backed — a rebind would detach the view)
        win = torch.cat([self.tail, x], dim=0)
        if self._backed:
            self.tail.copy_(win[1:])
        else:
            self.tail = win[1:]
        return win


class RingKV:
    """Fixed-capacity ring for one SWA layer's K/V — capacity = window (512).
    The window predicate kv_pos > q_pos - window (masking_utils
    sliding_window_overlay, B2.7) allows exactly the `window` most recent
    keys INCLUDING the current one, which after append is exactly the live
    ring content — so decode needs no mask at all."""

    def __init__(self, capacity, kv_heads, head_dim, device,
                 dtype=torch.bfloat16, backing=None):
        #1. slot for token at absolute position p is p % capacity.
        #   `backing` = (k, v) views of one SwaPool row (exp-0007): same
        #   [capacity, H, D] layout, so append/gather run this exact
        #   per-seq code on pooled storage; a prior occupant's stale rows
        #   are never read — gather stops at n, and the batched core masks
        #   slots whose position is >= n (model.attn_decode_batch_pooled)
        if backing is not None:
            self.k, self.v = backing
        else:
            self.k = torch.zeros(capacity, kv_heads, head_dim, device=device,
                                 dtype=dtype)
            self.v = torch.zeros_like(self.k)
        self.n = 0  # total tokens ever appended

    def append(self, k, v):
        """Append T tokens (k/v [T, kv_heads, head_dim]) at absolute
        positions n..n+T-1; only the last `capacity` can survive."""
        cap = self.k.shape[0]
        T = k.shape[0]
        #1. absolute positions of the incoming rows; drop rows a same-call
        #   later row would overwrite (T > cap only happens in prefill)
        pos = torch.arange(self.n, self.n + T, device=self.k.device)
        if T > cap:
            k, v, pos = k[-cap:], v[-cap:], pos[-cap:]
        #2. scatter into ring slots (positions are distinct mod cap here)
        slots = pos % cap
        self.k[slots] = k
        self.v[slots] = v
        self.n += T

    def gather(self):
        """Live entries in CHRONOLOGICAL order: (k, v, positions) with
        k/v [m, kv_heads, head_dim], positions [m] absolute (int64)."""
        cap = self.k.shape[0]
        m = min(self.n, cap)
        #1. oldest live token is n-m; chronological order keeps the @V
        #   accumulation order closest to the prefill computation
        pos = torch.arange(self.n - m, self.n, device=self.k.device)
        slots = pos % cap
        return self.k[slots], self.v[slots], pos


class PagedKV:
    """Paged KV for one GLOBAL layer: fixed page pool, PAGE_SIZE tokens per
    page, per-sequence page table (single sequence in B3.1 scope). Logical
    token position p lives in table[p // page_size] at offset p % page_size;
    pages are handed out from a caller-supplied free order, so physical
    placement is non-contiguous — gather() is real indirection."""

    def __init__(self, num_pages, page_size, kv_heads, head_dim, device,
                 dtype=torch.bfloat16, free_order=None, backing=None):
        #1. page pools, viewed flat [num_pages*page_size, H, D] for I/O.
        #   `backing` = (kp, vp, free, table_dev_row) from one GlobalPool
        #   (exp-0007): pool + free list SHARED by all co-resident
        #   sequences (page ids keep placements disjoint; placement never
        #   changes gathered values), table_dev_row a device-side mirror
        #   of self.table so the batched decode core indexes pages without
        #   the per-row _flat_slots .tolist() sync (exp-0005 finding)
        if backing is not None:
            self.kp, self.vp, self.free, self.table_dev_row = backing
        else:
            self.kp = torch.zeros(num_pages, page_size, kv_heads, head_dim,
                                  device=device, dtype=dtype)
            self.vp = torch.zeros_like(self.kp)
            self.free = list(free_order) if free_order is not None else list(
                range(num_pages))
            assert sorted(self.free) == list(range(num_pages)), \
                "bad free order"
            self.table_dev_row = None
        self.page_size = page_size
        self.table = []  # allocated page ids, token order
        self.n = 0       # tokens stored

    def _flat_slots(self, pos):
        #1. logical positions -> flat pool rows through the page table
        ps = self.page_size
        page_ids = torch.tensor([self.table[i] for i in
                                 torch.div(pos, ps, rounding_mode="floor")
                                 .tolist()], device=pos.device)
        return page_ids * ps + pos % ps

    def ensure_pages(self, last_page):
        """Allocate pages until the table covers `last_page` — free-list
        pop(0) order, deterministic given the schedule — mirroring new ids
        into table_dev_row when pool-backed (the batched decode core reads
        the device table, model.attn_decode_batch_pooled)."""
        #1. pop pages in free-list order; remember where the new ids start
        first_new = len(self.table)
        while len(self.table) <= last_page:
            self.table.append(self.free.pop(0))
        #2. mirror the new entries on-device for batched append/gather
        if self.table_dev_row is not None and len(self.table) > first_new:
            self.table_dev_row[first_new:len(self.table)] = torch.tensor(
                self.table[first_new:], device=self.table_dev_row.device)

    def append(self, k, v):
        """Append T tokens (k/v [T, kv_heads, head_dim]) at logical
        positions n..n+T-1, allocating pages from the free list on demand."""
        T = k.shape[0]
        #1. allocate pages to cover the new last position
        self.ensure_pages((self.n + T - 1) // self.page_size)
        #2. scatter rows through the table
        pos = torch.arange(self.n, self.n + T, device=self.kp.device)
        slots = self._flat_slots(pos)
        H, D = self.kp.shape[2], self.kp.shape[3]
        self.kp.view(-1, H, D)[slots] = k
        self.vp.view(-1, H, D)[slots] = v
        self.n += T

    def gather(self):
        """All stored tokens in logical order: (k, v, positions [0..n-1])."""
        pos = torch.arange(self.n, device=self.kp.device)
        slots = self._flat_slots(pos)
        H, D = self.kp.shape[2], self.kp.shape[3]
        return self.kp.view(-1, H, D)[slots], self.vp.view(-1, H, D)[slots], pos


class SwaPool:
    """exp-0007: batch-slot pool for one SWA layer's ring KV — row s is
    the ring of the sequence occupying scheduler slot s, so the batched
    decode core appends all B rows with ONE index_put_ and fetches all
    rings with ONE index gather instead of B per-row op chains. LayerKV
    hands RingKV views of rows, so the per-seq prefill/append code path
    is unchanged (and bitwise: same-shape tensors, same ops).

    Row max_batch is the SCRATCH slot (exp-0012): the CUDA-graphed decode
    step always runs at B = max_batch, and rows without a live sequence
    all point here. Dead rows are configured identically (prev token 0,
    pos 0, this slot), so their colliding index_put_ writes carry
    identical bytes — scratch content stays a pure function of the
    schedule and never feeds a live row."""

    def __init__(self, max_batch, window, kv_heads, head_dim, device,
                 dtype=torch.bfloat16):
        #1. [slot, ring_slot, H, D] — max_batch live rows + 1 scratch row
        self.k = torch.zeros(max_batch + 1, window, kv_heads, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)


class GlobalPool:
    """exp-0007: shared page pool for one GLOBAL layer — max_batch *
    num_pages pages, so total capacity equals max_batch per-seq PagedKVs
    exactly. One shared free list (pop(0)/extend order is a pure function
    of the schedule; placement never changes gathered values) and a
    device-side page-table mirror [max_batch, num_pages] so the batched
    decode core computes flat slots on-GPU — deleting the per-row
    _flat_slots .tolist() sync exp-0005 flagged (2 x 11 layers x B per
    step)."""

    def __init__(self, max_batch, num_pages, page_size, kv_heads, head_dim,
                 device, dtype=torch.bfloat16):
        #1. one pool; page ids span every slot's allocations, plus ONE
        #   scratch page (id `total`) for the graph path's dead rows
        #   (exp-0012, SwaPool docstring) — never on the free list
        total = max_batch * num_pages
        self.kp = torch.zeros(total + 1, page_size, kv_heads, head_dim,
                              device=device, dtype=dtype)
        self.vp = torch.zeros_like(self.kp)
        self.page_size = page_size
        self.free = list(range(total))
        #2. table mirror; entries beyond a sequence's live table are stale
        #   — never read (flat-slot gather is masked to positions < n).
        #   Row max_batch is the scratch slot: every entry points at the
        #   scratch page, so any dead-row position lands there
        self.table_dev = torch.zeros(max_batch + 1, num_pages,
                                     dtype=torch.long, device=device)
        self.table_dev[max_batch].fill_(total)


class SconvPool:
    """exp-0008: batch-slot pool for one layer's four sconv-site tails —
    row s holds the tails of the sequence in scheduler slot s, so the
    batched decode step rolls all B windows with ONE gather + ONE
    index_put_ per site (model.sconv_decode_batch_pooled) instead of B
    per-row cat/rebind chains — the copy_/cat storm exp-0005 counted
    (34,361 copy_ + 17,290 cat per step at B=64). SconvRing views of
    rows keep the per-seq prefill seed/step code on the same storage;
    rows are zeroed when (re)bound (SconvRing.__init__) because the conv
    window reads every tail row — stale occupants must be erased, not
    masked."""

    def __init__(self, max_batch, kernel, kv_ch, hidden, device,
                 dtype=torch.bfloat16):
        #1. [slot, kernel-1, C] per site — max_batch rings + the scratch
        #   row (exp-0012, SwaPool docstring)
        self.kt = torch.zeros(max_batch + 1, kernel - 1, kv_ch,
                              device=device, dtype=dtype)
        self.vt = torch.zeros_like(self.kt)
        self.at = torch.zeros(max_batch + 1, kernel - 1, hidden,
                              device=device, dtype=dtype)
        self.mt = torch.zeros_like(self.at)


class LayerKV:
    """One decoder layer's full recurrent state: the KV cache (PagedKV on
    the 11 global layers, RingKV(window) on the 55 SWA layers) + the four
    sconv rings (k/v sites carry kv_heads*head_dim channels; attn-out and
    mlp-out sites carry `hidden`). mc is the verified ModelConfig.
    `pool` + `slot` (exp-0007, scheduler admission) back the cache on the
    layer's batch pool — numerics and code paths identical, storage
    shared so the batched decode core can index it."""

    def __init__(self, mc, layer_idx, device, num_pages=None,
                 free_order=None, dtype=torch.bfloat16, pool=None,
                 slot=None):
        #1. layer-type split drives cache kind and KV head count (config.h)
        self.is_global = layer_idx in mc.global_layers
        self.slot = slot
        kv_heads = mc.g_kv_heads if self.is_global else mc.s_kv_heads
        if self.is_global:
            if pool is not None:
                self.cache = PagedKV(num_pages, pool.page_size, kv_heads,
                                     mc.head_dim, device, dtype,
                                     backing=(pool.kp, pool.vp, pool.free,
                                              pool.table_dev[slot]))
            else:
                assert num_pages is not None, \
                    "global layer needs a page budget"
                self.cache = PagedKV(num_pages, PAGE_SIZE, kv_heads,
                                     mc.head_dim, device, dtype, free_order)
        elif pool is not None:
            self.cache = RingKV(mc.window, kv_heads, mc.head_dim, device,
                                dtype, backing=(pool.k[slot], pool.v[slot]))
        else:
            self.cache = RingKV(mc.window, kv_heads, mc.head_dim, device,
                                dtype)
        #2. sconv raw-input tails (B2.7: the 3 raw inputs are state the
        #   post-conv KV entries do NOT carry); pool-backed when the KV
        #   pool carries a SconvPool (exp-0008: scheduler.init_pools)
        kvd = kv_heads * mc.head_dim
        sp = getattr(pool, "sconv", None) if pool is not None else None
        if sp is not None:
            self.k_ring = SconvRing(kvd, mc.sconv_k, device, dtype,
                                    backing=sp.kt[slot])
            self.v_ring = SconvRing(kvd, mc.sconv_k, device, dtype,
                                    backing=sp.vt[slot])
            self.attn_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype,
                                       backing=sp.at[slot])
            self.mlp_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype,
                                      backing=sp.mt[slot])
        else:
            self.k_ring = SconvRing(kvd, mc.sconv_k, device, dtype)
            self.v_ring = SconvRing(kvd, mc.sconv_k, device, dtype)
            self.attn_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype)
            self.mlp_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype)
