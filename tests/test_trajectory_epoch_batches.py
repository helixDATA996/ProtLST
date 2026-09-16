import pytest

from prot_lst.scripts.vae_trajectory.train_trajectory_vae import shuffled_epoch_batches


def test_epoch_batches_cover_every_record_once():
    batches = shuffled_epoch_batches(size=11, batch_size=3, seed=17)
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == 4
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_each_epoch_reshuffles_without_changing_coverage():
    first = shuffled_epoch_batches(size=20, batch_size=4, seed=17)
    second = shuffled_epoch_batches(size=20, batch_size=4, seed=18)

    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(20))
    assert sorted(index for batch in second for index in batch) == list(range(20))


def test_step_cap_shortens_epoch_batch_total():
    batches = shuffled_epoch_batches(size=20, batch_size=3, seed=17, max_batches=2)

    assert len(batches) == 2
    assert sum(map(len, batches)) == 6


@pytest.mark.parametrize("kwargs", [
    {"size": -1, "batch_size": 2, "seed": 17},
    {"size": 8, "batch_size": 0, "seed": 17},
    {"size": 8, "batch_size": 2, "seed": 17, "max_batches": -1},
])
def test_invalid_batch_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        shuffled_epoch_batches(**kwargs)
