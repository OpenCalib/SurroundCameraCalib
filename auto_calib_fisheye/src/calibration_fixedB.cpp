/*
    back camera fixed
*/
#include <math.h>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <vector>
#include "optimizer.h"
#include "texture_extractor.h"
#include "transform_util.h"

using namespace cv;
using namespace std;

double during_bev;
double during_compute_error;
double during_wrap;
string prefix, suffix;

// 3度|1cm
double CameraOptimization_sp_3_1(Optimizer &opt, string cameraType)
{
    std::chrono::_V2::steady_clock::time_point end_calib_ =
        chrono::steady_clock::now();
    ;
    double during_calib_ = 0;
    if (opt.coarse_flag)
    {
        cout << "**************************************1st*********************"
                "*******************"
             << endl;
        opt.phase      = 1;
        int thread_num = 7;
        vector<thread> threads(thread_num);
        auto start_calib = chrono::steady_clock::now();
        if (cameraType == "right")
        {
            vector<double> var(6, 0);
            opt.max_right_loss = opt.cur_right_loss =
                opt.CostFunction(var, "right", opt.extrinsic_right);
            int iter_nums = 100000;
            threads[0]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[1]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[2]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[3]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[4]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[5]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[6]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
        }
        else if (cameraType == "left")
        {
            vector<double> var(6, 0);
            opt.max_left_loss = opt.cur_left_loss =
                opt.CostFunction(var, "left", opt.extrinsic_left);
            int iter_nums = 100000;
            threads[0]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[1]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[2]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[3]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[4]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[5]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[6]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
        }
        else if (cameraType == "front")
        {
            vector<double> var(6, 0);
            opt.max_front_loss = opt.cur_front_loss =
                opt.CostFunction(var, "front", opt.extrinsic_front);
            int iter_nums = 100000;
            threads[0]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[1]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[2]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[3]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[4]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[5]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
            threads[6]    = thread(&Optimizer::random_search_params, &opt,
                                   iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                   -0.01, 0.01, -0.01, 0.01, cameraType);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads[i].join();
        }
        end_calib_ = chrono::steady_clock::now();
        during_calib_ =
            std::chrono::duration<double>(end_calib_ - start_calib).count();
        cout << "time:" << during_calib_ << endl;
        if (cameraType == "left")
        {
            opt.show("left", prefix + "/after_left_calib1.png");
            cout << "luminorsity loss before pre opt:" << opt.max_left_loss
                 << endl;
            cout << "luminorsity loss after pre opt:" << opt.cur_left_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.extrinsic_left_opt << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[0])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_bl:" << opt.ncoef_bl;
        }
        else if (cameraType == "front")
        {
            opt.show("front", prefix + "/after_front_calib1.png");
            cout << "luminorsity loss before pre opt:" << opt.max_front_loss
                 << endl;
            cout << "luminorsity loss after pre opt:" << opt.cur_front_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.extrinsic_front_opt << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[2])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_fl:" << opt.ncoef_fl;
            cout << endl << "ncoef_fr:" << opt.ncoef_fr;
        }
        else if (cameraType == "right")
        {
            opt.show("right", prefix + "/after_right_calib1.png");
            cout << "luminorsity loss before pre opt:" << opt.max_right_loss
                 << endl;
            cout << "luminorsity loss after pre opt:" << opt.cur_right_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.extrinsic_right_opt << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[1])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_br:" << opt.ncoef_br;
        }
        cout << endl;
    }
    else
    {
        opt.extrinsic_right_opt  = opt.extrinsic_right;
        opt.extrinsic_left_opt   = opt.extrinsic_left;
        opt.extrinsic_front_opt  = opt.extrinsic_front;
        opt.extrinsic_behind_opt = opt.extrinsic_behind;
    }

    cout << "**************************************2nd*************************"
            "***************"
         << endl;
    opt.phase = 2;
    double cur_right_loss_, cur_left_loss_, cur_front_loss_;
    int thread_num_ = 6;
    int iter_nums_  = 50000;
    vector<thread> threads_(thread_num_);
    if (cameraType == "right")
    {
        vector<double> var(6, 0);
        cur_right_loss_ = opt.cur_right_loss =
            opt.CostFunction(var, "right", opt.extrinsic_right_opt);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, cameraType);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, cameraType);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
    }
    else if (cameraType == "left")
    {
        vector<double> var(6, 0);
        cur_left_loss_ = opt.cur_left_loss =
            opt.CostFunction(var, "left", opt.extrinsic_left_opt);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, cameraType);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, cameraType);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
    }
    else if (cameraType == "front")
    {
        vector<double> var(6, 0);
        cur_front_loss_ = opt.cur_front_loss =
            opt.CostFunction(var, "front", opt.extrinsic_front_opt);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, cameraType);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, cameraType);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, cameraType);
    }
    for (int i = 0; i < thread_num_; i++)
    {
        threads_[i].join();
    }
    auto end_calib__ = chrono::steady_clock::now();
    double during_calib__ =
        std::chrono::duration<double>(end_calib__ - end_calib_).count();
    cout << "time:" << during_calib__ << endl;
    if (cameraType == "left")
    {
        opt.show("left", prefix + "/after_left_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_left_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_left_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_left_opt << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[0])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_bl:" << opt.ncoef_bl;
    }
    else if (cameraType == "front")
    {
        opt.show("front", prefix + "/after_front_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_front_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_front_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_front_opt << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[2])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_fl:" << opt.ncoef_fl;
        cout << endl << "ncoef_fr:" << opt.ncoef_fr;
    }
    else if (cameraType == "right")
    {
        opt.show("right", prefix + "/after_right_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_right_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_right_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_right_opt << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[1])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_br:" << opt.ncoef_br;
    }
    cout << endl;

    cout << "**************************************3rd*************************"
            "***************"
         << endl;
    opt.phase = 3;
    double cur_right_loss__, cur_left_loss__, cur_front_loss__;
    int thread_num__ = 3;
    int iter_nums__  = 20000;
    vector<thread> threads__(thread_num__);
    if (cameraType == "right")
    {
        vector<double> var(6, 0);
        cur_right_loss__ = opt.cur_right_loss =
            opt.CostFunction(var, "right", opt.extrinsic_right_opt);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, cameraType);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
    }
    else if (cameraType == "left")
    {
        vector<double> var(6, 0);
        cur_left_loss__ = opt.cur_left_loss =
            opt.CostFunction(var, "left", opt.extrinsic_left_opt);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, cameraType);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
    }
    else if (cameraType == "front")
    {
        vector<double> var(6, 0);
        cur_front_loss__ = opt.cur_front_loss =
            opt.CostFunction(var, "front", opt.extrinsic_front_opt);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, cameraType);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, cameraType);
    }
    for (int i = 0; i < thread_num__; i++)
    {
        threads__[i].join();
    }
    auto end_calib___ = chrono::steady_clock::now();
    double during_calib___ =
        std::chrono::duration<double>(end_calib___ - end_calib__).count();
    cout << "time:" << during_calib___ << endl;
    if (cameraType == "left")
    {
        opt.show("left", prefix + "/after_left_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_left_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_left_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_left_opt << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.extrinsic_left_opt.block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[0])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GL_opt.png",opt.imgl_bev_rgb);
    }
    else if (cameraType == "front")
    {
        opt.show("front", prefix + "/after_front_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_front_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_front_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_front_opt << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.extrinsic_front_opt.block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[2])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GB_opt.png",opt.imgb_bev_rgb);
    }
    else if (cameraType == "right")
    {
        opt.show("right", prefix + "/after_right_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_right_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_right_loss << endl;
        cout << "extrinsic after opt:" << endl
             << opt.extrinsic_right_opt << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.extrinsic_right_opt.block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[1])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GR_opt.png",opt.imgr_bev_rgb);
    }
    cout << endl
         << cameraType << " calibration time: "
         << during_calib_ + during_calib__ + during_calib___ << "s" << endl;
    return during_calib_ + during_calib__ + during_calib___;
}

int main()
{
    // camera_model:0-fisheye;1-Ocam;2-pinhole
    int camera_model = 0;

    // if add random disturbance to initial pose
    int flag_add_disturbance = 0;

    /*
    solution model :
        1.pure gray pipeline in three phase of optimization :
    solution_model_="gray" 2.(default)Adpative Threshold Binarization in first
    phase and pure gray in the 2nd&3rd phase of optimization:
            solution_model_="atb+gray"
        3.pure Adpative Threshold Binarization in all three phase of
    optimization: solution_model_="atb"
    */
    string solution_model_ = "gray";

    // if add road semantic segmentation when in texture extraction process to
    // improve accuracy
    int add_semantic_segmentation_left  = 1;
    int add_semantic_segmentation_right = 1;
    int add_semantic_segmentation_back  = 1;

    /*
    read surround frames
    In our demo,imgs1~2 are the surround images captured by a tour
    car(fisheye,Fov=195), imgs3~5 are the surround images captured in Carla
    engine(pinhole,Fov=125) imgs6 are the surround images captured by a car in
    the calibration room(fisheye,Fov=195).
    */
    prefix   = "../imgs1";
    suffix   = ".jpg";
    Mat imgb = cv::imread(prefix + "/e34/image_0_1679544704585" + suffix);
    Mat imgf = cv::imread(prefix + "/e34/image_1_1679544704868" + suffix);
    Mat imgl = cv::imread(prefix + "/e34/image_2_1679544704614" + suffix);
    Mat imgr = cv::imread(prefix + "/e34/image_3_1679544704586" + suffix);

    // bev rows、cols
    int bev_rows = 1000, bev_cols = 1000;

    // if add coarse search(1st search)
    int coarse_search_flag = 1;

    // which data_set(common or fisheye camera)
    string data_set = "imgs1";

    // which camera fixed
    string fixed = "back";

    // initilize the optimizer
    Optimizer opt(&imgf, &imgl, &imgb, &imgr, camera_model, bev_rows, bev_cols,
                  fixed, coarse_search_flag, data_set, flag_add_disturbance,
                  prefix, solution_model_);

    // bev images before optimization
    Mat GF = opt.project_on_ground(
        imgf, opt.extrinsic_front, opt.intrinsic_front,
        opt.distortion_params_front, opt.KG, opt.brows, opt.bcols, opt.hf);
    Mat GB = opt.project_on_ground(
        imgb, opt.extrinsic_behind, opt.intrinsic_behind,
        opt.distortion_params_behind, opt.KG, opt.brows, opt.bcols, opt.hb);
    Mat GL = opt.project_on_ground(imgl, opt.extrinsic_left, opt.intrinsic_left,
                                   opt.distortion_params_left, opt.KG,
                                   opt.brows, opt.bcols, opt.hl);
    Mat GR = opt.project_on_ground(
        imgr, opt.extrinsic_right, opt.intrinsic_right,
        opt.distortion_params_right, opt.KG, opt.brows, opt.bcols, opt.hr);

    // imshow("GF",GF);
    // waitKey(0);
    // imwrite(prefix+"/GF.png",GF);
    // imshow("GB",GB);
    // waitKey(0);
    // imwrite(prefix+"/GB.png",GB);
    // imshow("GL",GL);
    // waitKey(0);
    // imwrite(prefix+"/GL.png",GL);
    // imshow("GR",GR);
    // waitKey(0);
    // imwrite(prefix+"/GR.png",GR);

    GF = opt.tail(GF, "f");
    imwrite(prefix + "/GF_tail.png", GF);
    imshow("GF", GF);
    waitKey(0);
    GB = opt.tail(GB, "b");
    imwrite(prefix + "/GB_tail.png", GB);
    imshow("GB", GB);
    waitKey(0);
    GL = opt.tail(GL, "l");
    imwrite(prefix + "/GL_tail.png", GL);
    imshow("GL", GL);
    waitKey(0);
    GR = opt.tail(GR, "r");
    imwrite(prefix + "/GR_tail.png", GR);
    imshow("GR", GR);
    waitKey(0);

    Mat bev_before = opt.generate_surround_view(GF, GL, GB, GR);
    imwrite(prefix + "/before_all_calib.png", bev_before);
    imshow("opt_before", bev_before);
    waitKey(0);

    // back left field texture extraction
    vector<double> size  = {opt.sizef, opt.sizel, opt.sizeb, opt.sizer};
    int exposure_flag_bl = 1;  // if add exposure solution
    extractor ext1(GB, GL, add_semantic_segmentation_back, exposure_flag_bl,
                   size);
    if (add_semantic_segmentation_back)
    {
        Mat mask_bl = imread(prefix + "/mask/road_mask_back.png");
        ext1.mask_ground.push_back(mask_bl);
    }
    ext1.Binarization();
    ext1.findcontours();
    opt.bl_pixels_texture = ext1.extrac_textures_and_save(
        prefix + "/texture_bl.png", prefix + "/bl.csv");
    if (ext1.exposure_flag && ext1.ncoef > 0.5)
    {
        opt.ncoef_bl = ext1.ncoef;
        // cout<<"ncoef_bl:"<<opt.ncoef_bl<<endl;
    }
    else
    {
        opt.ncoef_bl = 1;
        // cout<<"ncoef_bl:"<<opt.ncoef_bl<<endl;
    }
    Mat pG_bl = Mat::ones(3, opt.bl_pixels_texture.size(), CV_64FC1);
    for (int i = 0; i < opt.bl_pixels_texture.size(); i++)
    {
        pG_bl.at<double>(0, i) = opt.bl_pixels_texture[i].first.x;
        pG_bl.at<double>(1, i) = opt.bl_pixels_texture[i].first.y;
    }
    opt.pG_bl = pG_bl;
    Mat PG_bl = Mat::ones(4, opt.bl_pixels_texture.size(), CV_64FC1);
    PG_bl(cv::Rect(0, 0, opt.bl_pixels_texture.size(), 3)) =
        opt.eigen2mat(opt.KG.inverse()) * pG_bl * opt.hb;
    opt.PG_bl = PG_bl;

    // back right field texture extraction
    int exposure_flag_br = 1;  // if add exposure solution
    extractor ext2(GB, GR, add_semantic_segmentation_back, exposure_flag_br,
                   size);
    if (add_semantic_segmentation_back)
    {
        Mat mask_br = imread(prefix + "/mask/road_mask_back.png");
        ext2.mask_ground.push_back(mask_br);
    }
    ext2.Binarization();
    ext2.findcontours();
    opt.br_pixels_texture = ext2.extrac_textures_and_save(
        prefix + "/texture_br.png", prefix + "/br.csv");
    if (ext2.exposure_flag && ext2.ncoef > 0.5)
    {
        opt.ncoef_br = ext2.ncoef;
        // cout<<"ncoef_br:"<<opt.ncoef_br<<endl;
    }
    else
    {
        opt.ncoef_br = 1;
        // cout<<"ncoef_br:"<<opt.ncoef_br<<endl;
    }
    Mat pG_br = Mat::ones(3, opt.br_pixels_texture.size(), CV_64FC1);
    for (int i = 0; i < opt.br_pixels_texture.size(); i++)
    {
        pG_br.at<double>(0, i) = opt.br_pixels_texture[i].first.x;
        pG_br.at<double>(1, i) = opt.br_pixels_texture[i].first.y;
    }
    opt.pG_br = pG_br;
    Mat PG_br = Mat::ones(4, opt.br_pixels_texture.size(), CV_64FC1);
    PG_br(cv::Rect(0, 0, opt.br_pixels_texture.size(), 3)) =
        opt.eigen2mat(opt.KG.inverse()) * pG_br * opt.hb;
    opt.PG_br = PG_br;

    cout << "*********************************start "
            "right*************************************"
         << endl;
    double during1 = CameraOptimization_sp_3_1(opt, "right");

    cout << "*********************************start "
            "left**************************************"
         << endl;
    double during2 = CameraOptimization_sp_3_1(opt, "left");

    // front left texture extraction
    int exposure_flag_fl = 1;  // if add exposure solution
    extractor ext3(opt.imgl_bev_rgb, GF, add_semantic_segmentation_left,
                   exposure_flag_fl, size);
    if (add_semantic_segmentation_left)
    {
        Mat mask_fl = imread(prefix + "/mask/road_mask_left.png");
        ext3.mask_ground.push_back(mask_fl);
    }
    ext3.Binarization();
    ext3.findcontours();
    opt.fl_pixels_texture = ext3.extrac_textures_and_save(
        prefix + "/texture_fl.png", prefix + "/fl.csv");
    if (ext3.exposure_flag && ext3.ncoef > 0.5)
    {
        opt.ncoef_fl = ext3.ncoef;
        // cout<<"ncoef_fl:"<<opt.ncoef_fl<<endl;
    }
    else
    {
        opt.ncoef_fl = 1;
        // cout<<"ncoef_fl:"<<opt.ncoef_fl<<endl;
    }
    Mat pG_fl = Mat::ones(3, opt.fl_pixels_texture.size(), CV_64FC1);
    for (int i = 0; i < opt.fl_pixels_texture.size(); i++)
    {
        pG_fl.at<double>(0, i) = opt.fl_pixels_texture[i].first.x;
        pG_fl.at<double>(1, i) = opt.fl_pixels_texture[i].first.y;
    }
    opt.pG_fl = pG_fl;
    Mat PG_fl = Mat::ones(4, opt.fl_pixels_texture.size(), CV_64FC1);
    PG_fl(cv::Rect(0, 0, opt.fl_pixels_texture.size(), 3)) =
        opt.eigen2mat(opt.KG.inverse()) * pG_fl * opt.hf;
    opt.PG_fl = PG_fl;

    // front right field texture extraction
    int exposure_flag_fr = 1;  // if add exposure solution
    extractor ext4(opt.imgr_bev_rgb, GF, add_semantic_segmentation_right,
                   exposure_flag_fr, size);
    if (add_semantic_segmentation_right)
    {
        Mat mask_fr = imread(prefix + "/mask/road_mask_right.png");
        ext4.mask_ground.push_back(mask_fr);
    }
    ext4.Binarization();
    ext4.findcontours();
    opt.fr_pixels_texture = ext4.extrac_textures_and_save(
        prefix + "/texture_fr.png", prefix + "/fr.csv");
    if (ext4.exposure_flag && ext4.ncoef > 0.5)
    {
        opt.ncoef_fr = ext4.ncoef;
        cout << "ncoef_fr:" << opt.ncoef_fr << endl;
    }
    else
    {
        opt.ncoef_fr = 1;
    }
    Mat pG_fr = Mat::ones(3, opt.fr_pixels_texture.size(), CV_64FC1);
    for (int i = 0; i < opt.fr_pixels_texture.size(); i++)
    {
        pG_fr.at<double>(0, i) = opt.fr_pixels_texture[i].first.x;
        pG_fr.at<double>(1, i) = opt.fr_pixels_texture[i].first.y;
    }
    opt.pG_fr = pG_fr;
    Mat PG_fr = Mat::ones(4, opt.fr_pixels_texture.size(), CV_64FC1);
    PG_fr(cv::Rect(0, 0, opt.fr_pixels_texture.size(), 3)) =
        opt.eigen2mat(opt.KG.inverse()) * pG_fr * opt.hf;
    opt.PG_fr = PG_fr;

    cout << "*********************************start "
            "front***********************************"
         << endl;
    double during3 = CameraOptimization_sp_3_1(opt, "front");

    cout << "************************online calibration "
            "finished!!!**************************"
         << endl;
    cout << "total calibration time:" << during1 + during2 + during3 << "s"
         << endl;

    opt.SaveOptResult(prefix + "/after_all_calib.png");
}