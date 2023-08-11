# surround-camera_calib
surround-camera_calib is a calibration toolbox for surround view cameras or surround view fisheye cameras, which contains four tools, as shown in the table below. For more calibration codes, please refer to the link <a href="https://github.com/PJLab-ADG/SensorsCalibration" title="SensorsCalibration">SensorsCalibration</a>
<!-- CITATION -->

| calibration param |calibration type| calibration method | mannual calibration | auto calibration | usage documentation |
| :--------------: |:--------------:| :------------: | :--------------: | :------------: | :------------: |
| surround_cameras (fisheye) | extrinsic |  target-less    |    &#10004; |             |[manual_calib](manual_calib/README.md)|
| surround_cameras (fisheye) | extrinsic |  target-less    |             |  &#10004;  |[auto_calib_fisheye](auto_calib_fisheye/README.md)|
| surround_cameras            | extrinsic |  target-less   |             |  &#10004;  |[auto_calib](auto_calib/README.md)|
| surround_cameras            | extrinsic |  target        |             |  &#10004;  |[auto_calib_target](auto_calib_target/README.md)|

## Prerequisites
- Cmake
- opencv 3.4.5
- eigen 3

## Compile
```shell
# mkdir build
mkdir -p build && cd build
# build
cmake .. && make
```
## Run(our examples)
```shell
cd bin
#choose 1.Calibrate by fixing the front
./run_AVM_Calibration_F
#choose 2.Calibrate by fixing the back
./run_AVM_Calibration_B
```
 ## Customize(your examples)
 If you need to calibrate base on your data, you can follow as below steps: 
### Prepare your surround camera images and the extrinsics and the intrinsic parameters etc.(set in /src/optimizer.cpp)
  - set the extrinsics in class function---Optimizer::initializePose()
  - set the intrinsics in class function---Optimizer::initializeK()
  - set the fisheye distortion parameters in class function---Optimizer::initializeD()  Ps:if pinhole camera, set 0
  - set the BEV camera intrinsic and height(or you keep same as ours) in class function---Optimizer::initializeKG() and Optimizer::initializeHeight()
  - set front,left,right,back BEV image tail size in class function---Optimizer::initializetailsize()
### Set calibration model(set in /src/calibration_fixedF.cpp or /src/calibration_fixedB.cpp)
  - choose camera model(camera_model)：0-fisheye;1-Ocam;2-pinhole
  - if add extra disturbance on surround cameras(flag_add_disturbance)：1-add;0-not
  - choose phase solution model(solution_model_):
    - 1.pure gray pipeline in three phase of optimization:
            solution_model_="gray"
    - 2.(default)Adpative Threshold Binarization in first phase and pure gray in the 2nd&3rd phase of optimization:  
            solution_model_="gray+atb"
    - 3.pure Adpative Threshold Binarization in all three phase of optimization:
            solution_model_="atb"
 - (Optional) if add road semantic segmentation when in texture extraction process to improve accuracy(add_semantic_segmentation_front/left/right/back):1-add 0-not
   - if you want to add road semantic segmentation mask you need to provide road semantic segmentation mask.
 - choose camera fixed(fixed):"front" or "back"


    

## Citation
If you find this project useful in your research, please consider cite:
```
@misc{2305.16840,
Author = {Jixiang Li and Jiahao Pi and Guohang Yan and Yikang Li},
Title = {Automatic Surround Camera Calibration Method in Road Scene for Self-driving Car},
Year = {2023},
Eprint = {arXiv:2305.16840},
}
```

## Contact
If you have questions about this repo, please contact Yan Guohang (`yanguohang@pjlab.org.cn`). 
