#!/bin/bash
# Submit all 928 experiment jobs as two SLURM arrays (MaxArraySize=500)
set -e

BASEDIR=/geode2/home/u040/joshnunl/BigRed200/voting-optimization

echo "Submitting array 1 (jobs 1-500)..."
JID1=$(sbatch --parsable "$BASEDIR/submit_array.sh")
echo "  Job ID: $JID1"

echo "Submitting array 2 (jobs 501-928)..."
JID2=$(sbatch --parsable "$BASEDIR/submit_array2.sh")
echo "  Job ID: $JID2"

echo "Done. Monitor with: squeue -u $(whoami)"
