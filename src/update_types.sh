#!/bin/zsh

# Make sure Gauge backend server is running on port 8000
# This openapi_transform.py will get the relevant parts of the openapi.json file and save it to openapi_new.json
uv run gaugelab/data/scripts/openapi_transform.py > gaugelab/data/openapi_new.json

# Then, datamodel-codegen will generate the gauge_types.py file based on the schema in openapi_new.json.
datamodel-codegen --input gaugelab/data/openapi_new.json --output gaugelab/data/gauge_types.py --use-annotated

# Post-process the generated file to fix mutable defaults
uv run gaugelab/data/scripts/fix_default_factory.py gaugelab/data/gauge_types.py

# Remove the openapi_new.json file since it is no longer needed
rm gaugelab/data/openapi_new.json