import jury_d as jd


def test_bar_state_has_three_states():
    assert jd.bar_state({"mean": 0.95, "lo": 0.92, "hi": 0.98}) == "above"
    assert jd.bar_state({"mean": 0.60, "lo": 0.52, "hi": 0.68}) == "under"
    assert jd.bar_state({"mean": 0.88, "lo": 0.80, "hi": 0.96}) == "undecided"


def test_h67_verdict_needs_both_trained_deltas_at_the_bar():
    ok = {"mean": 0.92, "lo": 0.85, "hi": 0.99}
    low = {"mean": 0.50, "lo": 0.44, "hi": 0.57}
    assert jd.h67_verdict({"dbpedia": ok, "mixed": ok}) == "holds"
    assert jd.h67_verdict({"dbpedia": ok, "mixed": low}) == "fails"


def test_cells_are_the_four_h67_cells():
    assert [c[0] for c in jd.cells()] == ["unmasked", "untrained", "dbpedia", "mixed"]
    assert all(k in (0, 12) for _, _, k in jd.cells())
