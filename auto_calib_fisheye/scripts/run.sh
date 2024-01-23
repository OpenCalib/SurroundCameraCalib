#!/bin/bash

export LD_LIBRARY_PATH=/opt/opencv/opencv-3.4.5/lib

./scripts/build.sh || { echo '*********************** Build failed! ***********************' ; exit 1; }

./bin/run_AVM_Calibration_F /home/kiennt63/release/calib/autorc-test/output /home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/imgs1 output
