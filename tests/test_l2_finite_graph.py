from core.lppm.finite_graph import verify_finite_cobuchi


def test_exact_finite_l2_returns_safe_and_constructs_p1_p2_values():
    result = verify_finite_cobuchi(
        {
            "waiting": ("waiting", "debt"),
            "debt": ("recovered",),
            "recovered": ("recovered",),
        },
        initial_states=("waiting",),
        bad_states=("debt",),
        eta=0.25,
    )
    assert result.verdict == "SAFE"
    assert result.p1_verified and result.p2_verified
    assert result.max_bad_visits == 1
    assert result.values == {"waiting": 0.25, "debt": 0.25, "recovered": 0.0}


def test_exact_finite_l2_returns_violation_with_bad_lasso():
    result = verify_finite_cobuchi(
        {
            "start": ("safe",),
            "safe": ("bad",),
            "bad": ("safe",),
        },
        initial_states=("start",),
        bad_states=("bad",),
    )
    assert result.verdict == "VIOLATION"
    assert result.lasso_prefix == ("start", "safe", "bad")
    assert result.lasso_cycle[0] == result.lasso_cycle[-1] == "bad"
    assert "safe" in result.lasso_cycle


def test_exact_finite_l2_rejects_dead_end_graphs():
    try:
        verify_finite_cobuchi(
            {"start": ("terminal",), "terminal": ()},
            initial_states=("start",),
            bad_states=(),
        )
    except ValueError as error:
        assert "dead ends" in str(error)
    else:
        raise AssertionError("dead-end infinite graph was accepted")
