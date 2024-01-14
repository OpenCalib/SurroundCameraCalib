#!/bin/bash

cmake -S . -Bbuild -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -GNinja
ninja -C build -j8

cd bin
./run_AVM_Calibration_F


