"""Per-old-class replay sampler tests (spec §8)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import add_path, run_tests  # noqa: E402

add_path()

import numpy as np  # noqa: E402

from shd_cl.training.replay import replay_percentages, sample_replay  # noqa: E402


def test_percentage_grid():
    assert replay_percentages(100, 0, 11) == [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert replay_percentages(100, 0, 1) == [100]


def test_per_old_class_count_half_ratio():
    """n_new=10, r=0.5 -> each old class gets round(0.5*10)=5 samples."""
    rng = np.random.default_rng(0)
    new_y = np.zeros(10, dtype=np.int64) + 19          # 10 new-class samples
    old_y = np.repeat(np.arange(19), 8)                 # 19 old classes, 8 each
    plan = sample_replay(new_y, old_y, replay_ratio=0.5, rng=rng)
    assert plan.log["n_new_samples"] == 10
    assert plan.log["m_old_per_class"] == 5
    assert all(v == 5 for v in plan.log["per_class_replay_counts"].values())
    assert plan.log["total_old_replay"] == 5 * 19
    assert plan.log["total_cil_train"] == 10 + 5 * 19
    assert len(plan.new_indices) == 10
    assert len(plan.replay_indices) == 5 * 19


def test_ratio_zero_means_no_replay():
    rng = np.random.default_rng(0)
    new_y = np.zeros(10, dtype=np.int64)
    old_y = np.repeat(np.arange(3), 5)
    plan = sample_replay(new_y, old_y, replay_ratio=0.0, rng=rng)
    assert plan.log["total_old_replay"] == 0
    assert len(plan.replay_indices) == 0


def test_ratio_one_is_class_balanced():
    """r=1 -> ~n_new per old class (class-balanced joint training)."""
    rng = np.random.default_rng(0)
    new_y = np.zeros(6, dtype=np.int64) + 4
    old_y = np.repeat(np.arange(3), 20)
    plan = sample_replay(new_y, old_y, replay_ratio=1.0, rng=rng)
    assert plan.log["m_old_per_class"] == 6
    assert all(v == 6 for v in plan.log["per_class_replay_counts"].values())


def test_replacement_policies_when_insufficient():
    rng = np.random.default_rng(0)
    new_y = np.zeros(10, dtype=np.int64)
    old_y = np.repeat(np.arange(2), 3)                  # only 3 samples per old class
    # want m = round(1.0*10) = 10 > 3 available
    p_repl = sample_replay(new_y, old_y, replay_ratio=1.0, rng=rng,
                           policy="with_replacement_if_needed")
    assert p_repl.log["replacement_used"] is True
    assert all(v == 10 for v in p_repl.log["per_class_replay_counts"].values())

    p_cap = sample_replay(new_y, old_y, replay_ratio=1.0, rng=np.random.default_rng(0),
                          policy="cap_at_available")
    assert p_cap.log["replacement_used"] is False
    assert all(v == 3 for v in p_cap.log["per_class_replay_counts"].values())

    try:
        sample_replay(new_y, old_y, replay_ratio=1.0, rng=np.random.default_rng(0),
                      policy="error")
        raise AssertionError("expected ValueError for policy=error")
    except ValueError:
        pass


def test_n_new_cap():
    rng = np.random.default_rng(0)
    new_y = np.zeros(50, dtype=np.int64)
    old_y = np.repeat(np.arange(3), 40)
    plan = sample_replay(new_y, old_y, replay_ratio=0.5, rng=rng, n_new_cap=10)
    assert plan.log["n_new_samples"] == 10
    assert plan.log["m_old_per_class"] == 5


if __name__ == "__main__":
    print("test_replay_sampler")
    raise SystemExit(run_tests(globals()))
