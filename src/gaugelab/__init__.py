# Import key components that should be publicly accessible
from gaugelab.clients import client, together_client
from gaugelab.gauge_client import GaugeClient
from gaugelab.version_check import check_latest_version

check_latest_version()

__all__ = [
    # Clients
    "client",
    "together_client",
    "GaugeClient",
]
