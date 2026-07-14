#!/usr/bin/env bash
set -euo pipefail
# Run from the project root.  Each stage uses the SAME shared Actor by passing
# the previous policy bundle as the training resume source in your Playground
# installation.  Labels are updated only by certify.py after freezing policy.

PYTHON=${PYTHON:-python}
CFG=${CFG:-configs/default.json}

$PYTHON -m cli.prepare_project --xml assets/orange_bike_2kg_horizontal.xml --reference data/reference_jump.csv

echo "1) Build/train/certify Landing -> Recovery"
$PYTHON -m cli.build_candidates --phase landing --target 96 --bank artifacts/landing_candidates.pkl --config "$CFG"
$PYTHON -m cli.train --stage landing --bank artifacts/landing_candidates.pkl --config "$CFG" --run runs/landing --timesteps 1000000
$PYTHON -m cli.certify --phase landing --policy runs/landing/policy --candidate-bank artifacts/landing_candidates.pkl --output-bank artifacts/landing_tube.pkl
$PYTHON -m cli.audit --phase landing --policy runs/landing/policy --bank artifacts/landing_tube.pkl --output runs/landing/audit.json

echo "2) Flight -> certified Landing entry -> Recovery"
$PYTHON -m cli.build_candidates --phase flight --target 160 --bank artifacts/flight_candidates.pkl --config "$CFG"
$PYTHON -m cli.train --stage flight --bank artifacts/flight_candidates.pkl --downstream-bank artifacts/landing_tube.pkl --config "$CFG" --run runs/flight --resume runs/landing/policy --timesteps 1200000
$PYTHON -m cli.certify --phase flight --policy runs/flight/policy --candidate-bank artifacts/flight_candidates.pkl --downstream-bank artifacts/landing_tube.pkl --output-bank artifacts/flight_tube.pkl
$PYTHON -m cli.audit --phase flight --policy runs/flight/policy --bank artifacts/flight_tube.pkl --downstream-bank artifacts/landing_tube.pkl --output runs/flight/audit.json

echo "3) Takeoff ground states -> certified Flight entry -> Recovery"
$PYTHON -m cli.build_candidates --phase takeoff --target 180 --bank artifacts/takeoff_candidates.pkl --config "$CFG"
$PYTHON -m cli.train --stage takeoff --bank artifacts/takeoff_candidates.pkl --downstream-bank artifacts/flight_tube.pkl --config "$CFG" --run runs/takeoff --resume runs/flight/policy --timesteps 1500000
$PYTHON -m cli.certify --phase takeoff --policy runs/takeoff/policy --candidate-bank artifacts/takeoff_candidates.pkl --downstream-bank artifacts/flight_tube.pkl --output-bank artifacts/takeoff_tube.pkl
$PYTHON -m cli.audit --phase takeoff --policy runs/takeoff/policy --bank artifacts/takeoff_tube.pkl --downstream-bank artifacts/flight_tube.pkl --output runs/takeoff/audit.json

echo "4) Approach -> certified Takeoff entry -> Recovery"
$PYTHON -m cli.build_candidates --phase approach --target 160 --bank artifacts/approach_candidates.pkl --config "$CFG"
$PYTHON -m cli.train --stage approach --bank artifacts/approach_candidates.pkl --downstream-bank artifacts/takeoff_tube.pkl --config "$CFG" --run runs/approach --resume runs/takeoff/policy --timesteps 1800000
$PYTHON -m cli.certify --phase approach --policy runs/approach/policy --candidate-bank artifacts/approach_candidates.pkl --downstream-bank artifacts/takeoff_tube.pkl --output-bank artifacts/approach_tube.pkl
$PYTHON -m cli.audit --phase approach --policy runs/approach/policy --bank artifacts/approach_tube.pkl --downstream-bank artifacts/takeoff_tube.pkl --output runs/approach/audit.json

echo "5) Natural-start complete jump evaluation"
$PYTHON -m cli.evaluate --stage full --policy runs/approach/policy --episodes 200 --output runs/natural_start_evaluation.json
