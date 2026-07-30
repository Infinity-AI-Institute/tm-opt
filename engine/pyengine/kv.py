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

    def __init__(self, channels, kernel, device, dtype=torch.bfloat16):
        #1. tail[i] = raw input at backward distance kernel-1-i (tail[-1]
        #   is the most recent token); zeros = implicit conv padding
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
        win = torch.cat([self.tail, x], dim=0)
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
    is unchanged (and bitwise: same-shape tensors, same ops)."""

    def __init__(self, max_batch, window, kv_heads, head_dim, device,
                 dtype=torch.bfloat16):
        #1. [slot, ring_slot, H, D] — the same layout as max_batch RingKVs
        self.k = torch.zeros(max_batch, window, kv_heads, head_dim,
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
        #1. one pool; page ids span every slot's allocations
        total = max_batch * num_pages
        self.kp = torch.zeros(total, page_size, kv_heads, head_dim,
                              device=device, dtype=dtype)
        self.vp = torch.zeros_like(self.kp)
        self.page_size = page_size
        self.free = list(range(total))
        #2. table mirror; entries beyond a sequence's live table are stale
        #   — never read (flat-slot gather is masked to positions < n)
        self.table_dev = torch.zeros(max_batch, num_pages,
                                     dtype=torch.long, device=device)


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
        #   post-conv KV entries do NOT carry)
        kvd = kv_heads * mc.head_dim
        self.k_ring = SconvRing(kvd, mc.sconv_k, device, dtype)
        self.v_ring = SconvRing(kvd, mc.sconv_k, device, dtype)
        self.attn_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype)
        self.mlp_ring = SconvRing(mc.hidden, mc.sconv_k, device, dtype)
