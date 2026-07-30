"""exp-0012 CPU unit tests — the pure-math pieces of the CUDA-graph
decode path (no GPU needed):
  1. decode_batch_ctx_static == decode_batch_ctx on identical inputs,
     and its padded columns are dead (g_valid False) beyond Lg.
  2. searchsorted segment offsets == the cumsum(bincount) form they
     replace in moe_experts_packed (value-identical integers).
  3. The pools carry the scratch row/page (kv.py exp-0012 shapes).
Run: python -m engine.pyengine.tests.t_graphrun"""
import types

import torch

from engine.pyengine import kv as pkv
from engine.pyengine import model as pmodel


def t_ctx_static():
    #1. same inputs -> same tensors as the dynamic builder
    mc = types.SimpleNamespace(window=8)
    pos = [3, 9, 0, 21]
    slots = [2, 0, 3, 4]
    ps = 4
    dyn = pmodel.decode_batch_ctx(pos, slots, mc, "cpu", ps)
    pos_t = torch.tensor(pos)
    slots_t = torch.tensor(slots)
    Lg = max(pos) + 1
    st = pmodel.decode_batch_ctx_static(pos_t, slots_t, Lg, mc, ps)
    for k in ("pos", "slots", "ring_slot", "swa_valid", "swa_dist",
              "g_pageidx", "g_off", "g_valid", "g_dist"):
        assert torch.equal(dyn[k], st[k]), k
    assert dyn["B"] == st["B"] and dyn["Lg"] == st["Lg"]
    #2. padding beyond Lg: prefix identical, padded columns dead
    L_pad = 32
    stp = pmodel.decode_batch_ctx_static(pos_t, slots_t, L_pad, mc, ps)
    for k in ("g_pageidx", "g_off", "g_valid", "g_dist"):
        assert torch.equal(stp[k][..., :Lg], dyn[k]), k
    assert not stp["g_valid"][:, Lg:].any()
    #3. dead-row config (pos 0): only its own position is live
    assert stp["g_valid"][2].sum() == 1 and stp["swa_valid"][2].sum() == 1
    print("t_ctx_static OK")


def t_searchsorted_offs():
    #1. the replaced form, on adversarial cases: empty experts at both
    #   ends, all-one-expert, uniform
    E = 16
    for flat in (torch.tensor([0, 0, 3, 3, 3, 15]),
                 torch.tensor([5] * 7),
                 torch.arange(E).repeat_interleave(2),
                 torch.randint(0, E, (96,))):
        sorted_e, _ = torch.sort(flat, stable=True)
        ref = torch.zeros(E + 1, dtype=torch.int32)
        ref[1:] = torch.cumsum(torch.bincount(sorted_e, minlength=E),
                               0).to(torch.int32)
        got = torch.searchsorted(
            sorted_e, torch.arange(E + 1, dtype=sorted_e.dtype)).to(
            torch.int32)
        assert torch.equal(ref, got), (ref, got)
    print("t_searchsorted_offs OK")


def t_scratch_pools():
    #1. every pool grows exactly one scratch row; the global scratch
    #   table row points every entry at the scratch page
    mb, np_, ps = 4, 8, pkv.PAGE_SIZE
    sw = pkv.SwaPool(mb, 16, 2, 4, "cpu")
    assert sw.k.shape[0] == mb + 1
    gp = pkv.GlobalPool(mb, np_, ps, 2, 4, "cpu")
    total = mb * np_
    assert gp.kp.shape[0] == total + 1
    assert gp.table_dev.shape[0] == mb + 1
    assert (gp.table_dev[mb] == total).all()
    assert gp.free == list(range(total))          # scratch never free
    sc = pkv.SconvPool(mb, 4, 8, 16, "cpu")
    assert sc.kt.shape[0] == mb + 1 and sc.at.shape[0] == mb + 1
    print("t_scratch_pools OK")


if __name__ == "__main__":
    t_ctx_static()
    t_searchsorted_offs()
    t_scratch_pools()
    print("ALL OK")
