"""
`gaugelab` answer relevancy scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from gaugelab.scorers.api_scorer import APIScorerConfig
from gaugelab.constants import APIScorerType


class DerailmentScorer(APIScorerConfig):
    score_type: APIScorerType = APIScorerType.DERAILMENT
