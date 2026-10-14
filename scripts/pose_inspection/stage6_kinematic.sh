#!/bin/bash
# Stage 6 (kinematic): no-pruning runs of the shape priors, then the prune-threshold sweep of every region type.
cd "$(dirname "$0")/../.."
rm -rf results/pose_inspection/kinematic/*/prior_template results/pose_inspection/kinematic/*/prior_cad \
       results/pose_inspection/kinematic/*/prior_*_geoprune results/pose_inspection/kinematic/*/sweep_*
scripts/pose_inspection/run_kinematic_batch.sh "prior_template prior_cad" "B_construction A_warehouse"
scripts/pose_inspection/sweep_prune.sh
echo "stage6 kinematic finished $(date +%H:%M:%S)"
