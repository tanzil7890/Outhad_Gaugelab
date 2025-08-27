from gaugelab.scorers.api_scorer import APIScorerConfig
from gaugelab.constants import APIScorerType
from gaugelab.data import ExampleParams
from typing import List


class AnswerRelevancyScorer(APIScorerConfig):
    score_type: APIScorerType = APIScorerType.ANSWER_RELEVANCY
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
    ]
