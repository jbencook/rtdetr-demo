"""Holdout logic."""
import enum

import farmhash


class Holdout(enum.Enum):
    """Holdout enum."""

    train: str = "train"
    dev: str = "dev"
    test: str = "test"


def in_holdout(slug: str, holdout: Holdout) -> bool:
    """Return True if the file is in the holdout."""
    hash_digit = farmhash.hash64(slug) % 10
    if hash_digit < 8 and holdout == Holdout.train:
        return True
    elif hash_digit == 8 and holdout == Holdout.dev:
        return True
    elif hash_digit == 9 and holdout == Holdout.test:
        return True
    return False
