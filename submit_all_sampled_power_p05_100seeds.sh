#!/bin/bash
set -e

BASEDIR=/geode2/home/u040/joshnunl/BigRed200/voting-optimization

echo "Submitting 100 sampled p=0.5 seed jobs..."
JID=$(sbatch --parsable "$BASEDIR/submit_sampled_power_p05_100seeds.sh")
echo "  Array job ID: $JID"

echo "Submitting dependent consolidation job..."
CJID=$(sbatch --parsable --dependency=afterok:${JID} \
  "$BASEDIR/submit_consolidate_sampled_power_p05_100seeds.sh")
echo "  Consolidation job ID: $CJID"

echo "Done."
echo "Seed outputs: $BASEDIR/src/UROC/results/sampled_curves/scoring_power_p0.50_100seeds/seed_*.npz"
echo "Combined output: $BASEDIR/src/UROC/results/sampled_curves/scoring_power_p0.50_100seeds.npz"
