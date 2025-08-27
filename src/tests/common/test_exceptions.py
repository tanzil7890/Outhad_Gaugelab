import pytest
from gaugelab.common.exceptions import (
    MissingTestCaseParamsError,
    GaugeAPIError,
    InvalidJudgeModelError,
)


def test_missing_test_case_params_error():
    """Test that MissingTestCaseParamsError can be raised"""
    with pytest.raises(MissingTestCaseParamsError):
        raise MissingTestCaseParamsError()


def test_gauge_api_error():
    """Test GaugeAPIError message handling"""
    error_message = "API connection failed"
    try:
        raise GaugeAPIError(error_message)
    except GaugeAPIError as e:
        assert str(e) == error_message
        assert e.message == error_message


def test_invalid_judge_model_error():
    """Test InvalidJudgeModelError message handling"""
    error_message = "Invalid model: gpt-5"
    try:
        raise InvalidJudgeModelError(error_message)
    except InvalidJudgeModelError as e:
        assert str(e) == error_message
        assert e.message == error_message
