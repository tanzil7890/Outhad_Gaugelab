"""
`gaugelab` hallucination scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from gaugelab.scorers.api_scorer import APIScorerConfig
from gaugelab.constants import APIScorerType
from gaugelab.data import ExampleParams


class HallucinationScorer(APIScorerConfig):
    def __init__(self, threshold: float):
        super().__init__(
            threshold=threshold,
            score_type=APIScorerType.HALLUCINATION,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
                ExampleParams.CONTEXT,
            ],
        )

    @property
    def __name__(self):
        return "Hallucination"
