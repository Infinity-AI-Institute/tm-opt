"""B3.1: KV state. Three stores, parity semantics with vLLM's specs
(BASELINE_NOTES.md): paged global KV (page 16, 8 KV heads x hd128, 11 layers),
ring-512 SWA KV (16 KV heads, 55 layers), sconv ring (window 4, 4 streams/layer).
Correctness oracle: decode(N+1) == recompute-from-scratch across the 512 edge."""
#TODO(B3.1)
