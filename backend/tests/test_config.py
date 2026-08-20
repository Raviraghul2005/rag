from config.loader import load_config


def test_default_config_loads():
    config = load_config()
    assert config.chunking.active_strategy == "recursive_512"
    assert len(config.corpus.languages) == 14
    assert config.corpus.passages_per_language == 10000
    # Calibrated (scripts/calibrate_guardrails.py), not a hardcoded guess — see
    # config/default.yaml's comment for the sweep this value came from.
    assert config.guardrails.tau_abs == 0.03
