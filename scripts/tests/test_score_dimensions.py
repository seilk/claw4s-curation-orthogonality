"""Tests for score_dimensions.py — unit tests for each scoring function."""
import sys
from pathlib import Path

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_conciseness_concise_beats_hedging():
    from score_dimensions import score_conciseness

    concise = {"response": "The answer is 4."}
    hedging = {
        "response": "Well, I'm not entirely sure, but I think the answer might possibly be 4, "
        "although I could be wrong about that and it's hard to say for certain."
    }
    s_concise = score_conciseness(concise)
    s_hedging = score_conciseness(hedging)
    assert s_concise > s_hedging, f"Concise ({s_concise}) should beat hedging ({s_hedging})"
    assert 0 <= s_concise <= 1
    assert 0 <= s_hedging <= 1


def test_conciseness_empty():
    from score_dimensions import score_conciseness

    assert score_conciseness({"response": ""}) == 0.0


def test_info_density_repetitive_is_low():
    from score_dimensions import score_info_density

    repetitive = {"response": "hello " * 100}
    varied = {"response": "The quick brown fox jumps over the lazy dog. " * 5 + "Quantum mechanics describes nature at the smallest scales."}
    s_rep = score_info_density(repetitive)
    s_var = score_info_density(varied)
    assert s_var > s_rep, f"Varied ({s_var}) should beat repetitive ({s_rep})"
    assert 0 <= s_rep <= 1
    assert 0 <= s_var <= 1


def test_info_density_empty():
    from score_dimensions import score_info_density

    assert score_info_density({"response": ""}) == 0.0
    assert score_info_density({"response": "hi"}) == 0.0  # < 10 chars


def test_distinct_n():
    from score_dimensions import _distinct_n

    # All unique bigrams
    assert _distinct_n("a b c d e") > 0.9
    # All same bigrams
    assert _distinct_n("a a a a a") < 0.3
    # Too short
    assert _distinct_n("a") == 0.0


def test_build_judge_prompt():
    from score_dimensions import build_judge_prompt

    sample = {"instruction": "What is 2+2?", "response": "4"}
    prompt = build_judge_prompt(sample, "accuracy")
    assert "2+2" in prompt
    assert "4" in prompt
    assert "1-10" in prompt
