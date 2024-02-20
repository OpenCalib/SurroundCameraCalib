#!/bin/bash

in_dir=$1
out_dir=$2

export LD_LIBRARY_PATH=/opt/opencv/opencv-3.4.5/lib

for d in $(find ${in_dir}/* -type d); do
    echo "Running calibration for $d"
    timestamp=$(basename $d)
    mkdir -p ${out_dir}/${timestamp}
    ./bin/run_AVM_Calibration_F custom /home/kiennt63/release/calib/vf8-eco-bowl-16x18/output ${d} ${out_dir}/${timestamp}
done
