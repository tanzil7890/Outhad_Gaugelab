import pytest
from pydantic import ValidationError

from gaugelab.scorers.api_scorer import APIScorerConfig
from gaugelab.constants import APIScorerType


@pytest.fixture
def valid_scorer_params():
    return {"threshold": 0.8, "score_type": APIScorerType.FAITHFULNESS}


@pytest.mark.parametrize(
    "invalid_score_type",
    [
        "random string",  # string that isnt an enum
        123,  # integer
        None,  # None
        True,  # boolean
        ["faithfulness"],  # list
        {"type": "faithfulness"},  # dict
    ],
)
def test_gauge_scorer_invalid_score_type(invalid_score_type):
    """Test creating GaugeScorer with invalid score_type values"""
    with pytest.raises(ValidationError) as exc_info:
        APIScorerConfig(threshold=0.8, score_type=invalid_score_type)

    assert "Input should be" in str(exc_info.value)


def test_gauge_scorer_invalid_string_value():
    """Test creating GaugeScorer with invalid string value"""
    with pytest.raises(ValidationError):
        APIScorerConfig(threshold=0.8, score_type="INVALID_METRIC")


def test_gauge_scorer_threshold_validation():
    """Test threshold validation"""
    # Test float values
    scorer = APIScorerConfig(threshold=0.5, score_type=APIScorerType.FAITHFULNESS)
    assert scorer.threshold == 0.5

    # Test integer values (should be converted to float)
    scorer = APIScorerConfig(threshold=1, score_type=APIScorerType.FAITHFULNESS)
    assert scorer.threshold == 1.0

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        scorer = APIScorerConfig(threshold=1.5, score_type=APIScorerType.FAITHFULNESS)
