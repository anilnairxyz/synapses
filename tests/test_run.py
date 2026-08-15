"""CPU-only tests for run.py plumbing: log_spaced_epochs, list_checkpoints, cosine_warmup_lr."""

from __future__ import annotations

import pytest
import torch

from synapses.run import cosine_warmup_lr, list_checkpoints, log_spaced_epochs


def test_log_spaced_epochs_includes_powers_and_final():
    assert log_spaced_epochs(30) == {1, 2, 4, 8, 16, 30}
    assert log_spaced_epochs(1) == {1}
    # a power-of-two horizon: the final epoch coincides with a power and isn't duplicated
    assert log_spaced_epochs(16) == {1, 2, 4, 8, 16}


def test_list_checkpoints_sorted_and_parsed(tmp_path):
    ckpt = tmp_path / "run" / "ckpt"
    ckpt.mkdir(parents=True)
    for e in (2, 16, 1, 8, 4, 30):  # created out of order
        (ckpt / f"epoch_{e:04d}.pt").write_bytes(b"")
    (ckpt / "not_a_ckpt.txt").write_bytes(b"")  # ignored

    got = list_checkpoints(tmp_path / "run")
    assert [e for e, _ in got] == [1, 2, 4, 8, 16, 30]
    assert all(p.suffix == ".pt" for _, p in got)


def test_list_checkpoints_missing_dir(tmp_path):
    assert list_checkpoints(tmp_path / "nope") == []


def test_cosine_warmup_lr_ramps_then_decays():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = cosine_warmup_lr(opt, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert lrs[0] == 0.0  # step 0 of warmup
    assert lrs[9] < lrs[10]  # still ramping through warmup
    assert lrs[10] == pytest.approx(1.0, abs=0.05)  # peak just after warmup ~= base lr
    assert lrs[-1] < lrs[10]  # cosine decay brings it back down
    assert all(lr >= 0 for lr in lrs)
