from config.loader import load_config


def test_default_config_loads():
    config = load_config()
    assert config.chunking.active_strategy == "recursive_512"
    assert len(config.corpus.languages) == 14
    assert config.corpus.passages_per_language is None
    assert config.guardrails.tau_abs is None
