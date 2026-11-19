"""clsprobe.py: first-token label scores on classify, pure parts."""

import numpy as np
import pytest

import clsprobe as cp


def test_rotate_labels_moves_world_last_and_keeps_the_rest():
    p = "Classify this news item into one of: World, Sports, Business, Sci/Tech. Reply with the label only.\n\nText"
    out = cp.rotate_labels(p)
    assert "one of: Sports, Business, Sci/Tech, World. Reply" in out and out.endswith("\n\nText")


def test_rotate_labels_refuses_another_prompt():
    with pytest.raises(ValueError, match="label list"):
        cp.rotate_labels("no list here")


def test_article_span_is_what_differs_from_the_empty_prompt():
    empty = [1, 2, 3, 8, 9]
    ids = [1, 2, 3, 40, 41, 42, 8, 9]
    assert cp.article_span(ids, empty) == (3, 6)


def test_article_span_with_an_article_that_repeats_the_suffix():
    assert cp.article_span([1, 2, 8, 9, 8, 9], [1, 2, 8, 9]) == (2, 4)


def test_offset_share_is_one_for_a_constant_shift_and_near_zero_for_noise():
    constant = np.tile(np.array([1.0, -1.0, 0.5, -0.5]), (200, 1))
    assert cp.offset_share(constant) == pytest.approx(1.0)
    noise = np.random.default_rng(0).normal(size=(2000, 4))
    assert cp.offset_share(noise) < 0.02


def test_offset_share_ignores_a_per_item_level():
    x = np.tile(np.array([1.0, -1.0, 0.0, 0.0]), (10, 1)) + np.arange(10)[:, None]
    assert cp.offset_share(x) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("first", "world", "verdict"),
    [
        ({"a": 0.20, "b": 0.15, "c": 0.0}, {"a": -0.20, "b": -0.15, "c": 0.0}, "first_position"),
        ({"a": 0.20, "b": 0.0, "c": 0.0}, {"a": -0.20, "b": 0.0, "c": 0.0}, "not_first_position"),
        ({"a": 0.20, "b": 0.20, "c": 0.0}, {"a": 0.0, "b": 0.0, "c": 0.0}, "not_first_position"),
    ],
)
def test_rotation_verdict(first, world, verdict):
    assert cp.rotation_verdict(first, world) == verdict


def test_article_verdict_needs_ten_points_on_two_models():
    assert cp.article_verdict({"a": 0.10, "b": 0.12, "c": -0.3}) == "holds"
    assert cp.article_verdict({"a": 0.10, "b": 0.09, "c": 0.0}) == "fails"


def test_rotate_list_moves_the_first_label_last():
    p = "Classify this text into one of: a, b, c. Reply with the label only.\n\nT"
    assert "one of: b, c, a. Reply" in cp.rotate_list(p, ("a", "b", "c"))


def test_emotion_prompt_lists_the_labels_in_the_given_order():
    assert cp.emotion_prompt("hello").startswith(
        "Classify this text into one of: sadness, joy, love, anger, fear, surprise."
    )
    assert cp.emotion_prompt("hello").endswith("\n\nhello")


def test_emotion_cells_have_their_own_paths():
    assert cp.cell_path("gemma", 12, "x", "ends", False, "emotion").name == "clsprobe_emotion_gemma_k12_x_ends.jsonl"
    assert cp.cell_path("gemma", 0, "", "", True, "emotion").name == "clsprobe_emotion_gemma_k0_rot.jsonl"
    assert cp.cell_path("gemma", 12, "x", "ends", False).name == "clsprobe_gemma_k12_x_ends.jsonl"


@pytest.mark.parametrize(
    ("arm", "a", "a2", "expected"),
    [
        ({"mean": 0.95, "lo": 0.9, "hi": 1.0}, 0.70, 0.72, True),
        ({"mean": 0.88, "lo": 0.85, "hi": 0.91}, 0.70, 0.72, True),
        ({"mean": 0.80, "lo": 0.75, "hi": 0.85}, 0.70, 0.72, False),
        ({"mean": 0.92, "lo": 0.88, "hi": 0.96}, 0.70, 0.90, False),
    ],
)
def test_beats_noise_needs_the_bar_and_a_gain_above_the_seed_gap(arm, a, a2, expected):
    assert cp.beats_noise(arm, a, a2) is expected


def test_rotate_list_by_n_cycles_the_labels():
    p = "one of: a, b, c, d. Reply"
    assert "b, c, d, a." in cp.rotate_list(p, ("a", "b", "c", "d"), 1)
    assert "c, d, a, b." in cp.rotate_list(p, ("a", "b", "c", "d"), 2)
    assert "d, a, b, c." in cp.rotate_list(p, ("a", "b", "c", "d"), 3)
    assert cp.rotate_list(p, ("a", "b", "c", "d"), 0) == p


def test_cell_path_names_each_order_and_keeps_the_h38a_names():
    assert cp.cell_path("g", 12, "x", "ends", False).name == "clsprobe_g_k12_x_ends.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", True).name == "clsprobe_g_k12_x_ends_rot.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", 1).name == "clsprobe_g_k12_x_ends_rot.jsonl"
    assert cp.cell_path("g", 12, "x", "ends", 2).name == "clsprobe_g_k12_x_ends_rot2.jsonl"
    assert cp.cell_path("g", 0, "", "", 3).name == "clsprobe_g_k0_rot3.jsonl"


def test_find_spans_takes_the_first_occurrence_of_any_spelling():
    ids = [9, 1, 7, 2, 3, 7, 1, 5]
    spans = cp.find_spans(ids, {"A": [[1]], "B": [[8], [2, 3]]})
    assert spans == {"A": (1, 2), "B": (3, 5)}


def test_find_spans_refuses_a_missing_label():
    with pytest.raises(ValueError, match="B"):
        cp.find_spans([1, 2], {"B": [[8]]})


def test_rule_parts():
    assert cp.rule_parts("ends") == ("ends", 0)
    assert cp.rule_parts("labels2") == ("ends", 2)
    assert cp.rule_parts("article") == ("article", 0)


def test_h39_clauses():
    ends = {"a": {"mean": 0.60, "gap": 0.20}, "b": {"mean": 0.70, "gap": 0.10}, "c": {"mean": 0.90, "gap": 0.02}}
    fixed = {"a": {"mean": 0.75, "gap": 0.10}, "b": {"mean": 0.80, "gap": 0.05}, "c": {"mean": 0.88, "gap": 0.02}}
    assert cp.h39_collapse(fixed, ends) == "holds"
    assert cp.h39_gap(fixed, ends) == "holds"
    weak = {"a": {"mean": 0.65, "gap": 0.15}, "b": {"mean": 0.80, "gap": 0.20}, "c": {"mean": 0.90, "gap": 0.02}}
    assert cp.h39_collapse(weak, ends) == "fails"
    assert cp.h39_gap(weak, ends) == "fails"


def test_order_summary_uses_each_order():
    assert cp.order_summary([0.9, 0.7, 0.8, 0.8]) == pytest.approx({"mean": 0.8, "worst": 0.7, "gap": 0.2})


def test_offset_share_of_no_change_is_zero():
    assert cp.offset_share(np.zeros((5, 4))) == 0.0


def test_h40_needs_every_order_at_the_bar_on_every_model():
    good = {"mean": 0.95, "lo": 0.91, "hi": 0.99}
    by_ci = {"mean": 0.87, "lo": 0.82, "hi": 0.92}
    bad = {"mean": 0.80, "lo": 0.75, "hi": 0.85}
    assert cp.h40_verdict({"a": [good, by_ci], "b": [good, good]}) == {
        "verdict": "holds",
        "at_bar": {"a": True, "b": True},
    }
    assert cp.h40_verdict({"a": [good, bad], "b": [good, good]}) == {
        "verdict": "fails",
        "at_bar": {"a": False, "b": True},
    }


def _ci(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi}


def test_h63_gate_needs_every_known_seed_above_every_a_seed():
    a = [_ci(0.667, 0.6, 0.73), _ci(0.86, 0.81, 0.91)]
    assert cp.h63_gate(a, [_ci(0.93, 0.88, 0.98), _ci(0.96, 0.92, 1.0)]) is True
    assert cp.h63_gate(a + [_ci(0.95, 0.92, 0.99)], [_ci(0.915, 0.875, 0.955), _ci(0.96, 0.92, 1.0)]) is False


def test_h63_gate_fails_when_the_lowest_known_seed_misses_the_bar():
    assert cp.h63_gate([_ci(0.5, 0.4, 0.6)], [_ci(0.8, 0.75, 0.85), _ci(0.96, 0.92, 1.0)]) is False


def test_exact_rank_p_counts_label_splits():
    assert cp.exact_rank_p([4, 5, 6], [1, 2, 3]) == 1 / 20
    assert cp.exact_rank_p([1, 2, 3], [4, 5, 6]) == 1.0
    assert cp.exact_rank_p([1, 5], [2, 3, 4], two_sided=True) == 1.0


def test_exact_rank_p_with_six_against_six_has_floor_one_in_924():
    assert cp.exact_rank_p(list(range(7, 13)), list(range(1, 7))) == 1 / 924


def test_cell_path_marks_a_later_start_and_keeps_the_old_names():
    assert cp.cell_path("q", 12, "x", "ends", False, "", 300).name == "clsprobe_q_k12_x_ends.jsonl"
    assert cp.cell_path("q", 12, "x", "ends", True, "", 600).name == "clsprobe_q_k12_x_ends_rot_d600.jsonl"
    assert cp.cell_path("q", 0, "", "", False, "emotion", 600).name == "clsprobe_emotion_q_k0_d600.jsonl"


def test_emotion_window_moves_with_the_start():
    assert cp.emotion_window(300) == (0, 300)
    assert cp.emotion_window(600) == (300, 600)


def test_h68_reading_follows_the_fixed_rule():
    """Pairs are (p of A: dbrand above a, p of B: known above dbrand)."""
    both = ("cls_acc", "ag_rot")
    assert cp.h68_reading({k: (0.4, 0.01) for k in both}) == "data explains"
    assert cp.h68_reading({k: (0.01, 0.4) for k in both}) == "order explains"
    assert cp.h68_reading({k: (0.01, 0.01) for k in both}) == "both contribute"
    assert cp.h68_reading({k: (0.4, 0.4) for k in both}) == "inconclusive"
    assert cp.h68_reading({"cls_acc": (0.4, 0.01), "ag_rot": (0.4, 0.4)}) == "split"
    assert cp.h68_reading({"cls_acc": (0.4, 0.01), "ag_rot": (0.01, 0.4)}) == "split"


def test_paired_diff_is_trained_minus_untrained_in_points_on_matched_items():
    d = cp.paired_diff({1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0}, {1: 1.0, 2: 0.0, 3: 0.0, 5: 1.0})
    assert d["n"] == 3
    assert abs(d["mean"] - 100 / 3) < 1e-9
    assert d["lo"] <= d["mean"] <= d["hi"]
