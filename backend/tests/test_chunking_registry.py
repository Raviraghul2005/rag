import pytest

from app.chunking.registry import STRATEGY_NAMES, get_strategy, requires_encoder
from app.models.chunk import Document
from config.loader import load_config


def test_config_strategies_all_resolvable():
    # Guards against config listing a strategy name the registry doesn't implement.
    for name in load_config().chunking.strategies:
        assert name in STRATEGY_NAMES


def test_active_strategy_is_known():
    assert load_config().chunking.active_strategy in STRATEGY_NAMES


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_strategy("does_not_exist")


def test_encoder_bound_strategy_without_encoder_raises():
    with pytest.raises(ValueError):
        get_strategy("late_chunking")


@pytest.mark.parametrize("name", [n for n in STRATEGY_NAMES if not requires_encoder(n)])
def test_encoder_free_strategies_produce_chunks(name):
    doc = Document(doc_id="d1", text="भारत एक देश है। यहाँ भाषाएँ हैं।", language="hi", query_type=None)
    chunks = get_strategy(name).chunk(doc)
    assert chunks
    assert all(c.strategy for c in chunks)
