#!/bin/bash

export LD_LIBRARY_PATH=/opt/opencv/opencv-3.4.5/lib

./scripts/build.sh || { echo '*********************** Build failed! ***********************' ; exit 1; }

# ./bin/run_AVM_Calibration_F imgs1 /home/kiennt63/release/calib/autorc-test/output /home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/imgs1 reference

./bin/run_AVM_Calibration_F custom /home/kiennt63/release/calib/autorc-test/output /home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/custom output

# ./bin/run_AVM_Calibration_F custom /home/kiennt63/release/calib/vf8-eco-bowl-16x18/output /home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/vf8_eco vf8_output
