"""
`gaugelab` faithfulness scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from gaugelab.scorers.api_scorer import APIScorerConfig
from gaugelab.constants import APIScorerType
from gaugelab.data import ExampleParams
from typing import List


class FaithfulnessScorer(APIScorerConfig):
    score_type: APIScorerType = APIScorerType.FAITHFULNESS
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.RETRIEVAL_CONTEXT,
    ]
