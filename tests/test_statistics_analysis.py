from app.evaluation.statistics_analysis import check_normality, compare_three_groups, cohens_d


def test_check_normality_returns_expected_keys():
    result = check_normality([0.5, 0.6, 0.55, 0.57, 0.58])
    assert "p_value" in result
    assert "is_normal" in result


def test_compare_three_groups_returns_test_name():
    result = compare_three_groups(
        [0.5, 0.6, 0.55],
        [0.7, 0.72, 0.71],
        [0.4, 0.45, 0.43]
    )
    assert "test" in result


def test_cohens_d_returns_float():
    d = cohens_d([0.5, 0.6, 0.55], [0.7, 0.72, 0.71])
    assert isinstance(d, float)