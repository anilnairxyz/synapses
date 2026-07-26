"""CPU-only tests for run.py plumbing: log_spaced_epochs and list_checkpoints."""

from __future__ import annotations

from synapses.run import list_checkpoints, log_spaced_epochs


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
