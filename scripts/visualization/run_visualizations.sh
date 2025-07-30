#!/bin/bash
# Generate visualizations for all completed uTooth experiments

cd /home/gaetano/utooth

echo "=== Generating visualizations for all completed runs ==="

# List of completed experiments (excluding utooth_10f_v5 which is incomplete)
completed_runs=("utooth_5f_v1" "utooth_10f_v1" "utooth_10f_v2" "utooth_10f_v3" "utooth_10f_v4")

for run in "${completed_runs[@]}"; do
    echo "Processing $run..."
    python scripts/visualization/visualize_run.py "$run"
    if [ $? -eq 0 ]; then
        echo "✓ Success: $run"
    else
        echo "✗ Failed: $run"
    fi
done

echo ""
echo "=== Generated visualization files ==="
find outputs/visualizations -name "*.png" -type f -ls | wc -l | xargs echo "Total PNG files:"

echo ""
echo "=== Usage ==="
echo "View individual experiment visualizations:"
for run in "${completed_runs[@]}"; do
    echo "  outputs/visualizations/$run/${run}_analysis.png"
    echo "  outputs/visualizations/$run/${run}_summary_table.png"
done