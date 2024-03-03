/*
    front camera fixed
*/
#include <math.h>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <ctime>
#include <filesystem>
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
#include "config_parser.h"
#include "defines.h"
#include "generate_lut.h"
#include "image_processor_context.h"
#include "image_processor_cuda.h"
#include "optimizer.h"
#include "perception_config.pb.h"
#include "segment_image_processor_cuda.h"
#include "texture_extractor.h"
#include "transform_util.h"
#include "utils.h"

using namespace cv;
using namespace std;

double during_bev;
double during_compute_error;
double during_wrap;
string calib, input, output, extension;

/*
Deviable Hierarchical Search Optimization Based on Concurrent Mode,
note:
    The search scopes in all phase of roll,pitch,yaw,dx,dy,dz and iterations
could be customized based on your data.
*/
double CameraOptimization(Optimizer& opt, CamID camId)
{
    std::chrono::_V2::steady_clock::time_point end_calib_ =
        chrono::steady_clock::now();

    double during_calib_ = 0;
    if (opt.coarse_flag)
    {
        cout << "**************************************1st*********************"
                "*******************"
             << endl;
        opt.phase      = 1;
        int thread_num = 7;
        int iter_nums  = 100000;
        vector<thread> threads(thread_num);
        auto start_calib = chrono::steady_clock::now();
        if (camId == CamID::R)
        {
            vector<double> var(6, 0);
            opt.max_right_loss = opt.cur_right_loss =
                opt.CostFunction(var, CamID::R, opt.initExt[CamID::R]);

            threads[0] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[1] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[2] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[3] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[4] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[5] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[6] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
        }
        else if (camId == CamID::L)
        {
            vector<double> var(6, 0);
            opt.max_left_loss = opt.cur_left_loss =
                opt.CostFunction(var, CamID::L, opt.initExt[CamID::L]);

            threads[0] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[1] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[2] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[3] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[4] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[5] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[6] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
        }
        else if (camId == CamID::B)
        {
            vector<double> var(6, 0);
            opt.max_behind_loss = opt.cur_behind_loss =
                opt.CostFunction(var, CamID::B, opt.initExt[CamID::B]);

            threads[0] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[1] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 0, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[2] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, 0, 3, -3, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[3] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 0, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[4] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, 0, 3, -3, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[5] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, -3, 0, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
            threads[6] = thread(&Optimizer::random_search_params, &opt,
                                iter_nums, -3, 3, -3, 3, 0, 3, -0.01, 0.01,
                                -0.01, 0.01, -0.01, 0.01, camId);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads[i].join();
        }
        end_calib_ = chrono::steady_clock::now();
        during_calib_ =
            std::chrono::duration<double>(end_calib_ - start_calib).count();
        cout << "time:" << during_calib_ << endl;
        if (camId == CamID::L)
        {
            // opt.show(CamID::L,prefix+"/after_left_calib1.png");
            cout << "luminorsity loss before 1st opt:" << opt.max_left_loss
                 << endl;
            cout << "luminorsity loss after 1st opt:" << opt.cur_left_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.optExt[CamID::L] << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[0])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_fl:" << opt.ncoef_fl;
        }
        else if (camId == CamID::B)
        {
            // opt.show(CamID::B,prefix+"/after_behind_calib1.png");
            cout << "luminorsity loss before 1st opt:" << opt.max_behind_loss
                 << endl;
            cout << "luminorsity loss after 1st opt:" << opt.cur_behind_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.optExt[CamID::B] << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[2])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_bl:" << opt.ncoef_bl;
            cout << endl << "ncoef_br:" << opt.ncoef_br;
        }
        else if (camId == CamID::R)
        {
            // opt.show(CamID::R,prefix+"/after_right_calib1.png");
            cout << "luminorsity loss before 1st opt:" << opt.max_right_loss
                 << endl;
            cout << "luminorsity loss after 1st opt:" << opt.cur_right_loss
                 << endl;
            cout << "extrinsic after pre opt:" << endl
                 << opt.optExt[CamID::R] << endl;
            cout << "best search parameters:" << endl;
            for (auto e : opt.bestVal_[1])
                cout << fixed << setprecision(6) << e << " ";
            cout << endl << "ncoef_fr:" << opt.ncoef_fr;
        }
        cout << endl;
    }
    else
    {
        opt.optExt[R] = opt.initExt[R];
        opt.optExt[L] = opt.initExt[L];
        opt.optExt[F] = opt.initExt[F];
        opt.optExt[B] = opt.initExt[B];
    }

    cout << "**************************************2nd*************************"
            "***************"
         << endl;
    opt.phase = 2;
    double cur_right_loss_, cur_left_loss_, cur_behind_loss_;
    int thread_num_ = 6;
    int iter_nums_  = 50000;
    vector<thread> threads_(thread_num_);
    if (camId == CamID::R)
    {
        vector<double> var(6, 0);
        cur_right_loss_ = opt.cur_right_loss =
            opt.CostFunction(var, CamID::R, opt.optExt[CamID::R]);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, camId);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, camId);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
    }
    else if (camId == CamID::L)
    {
        vector<double> var(6, 0);
        cur_left_loss_ = opt.cur_left_loss =
            opt.CostFunction(var, CamID::L, opt.optExt[CamID::L]);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, camId);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, camId);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
    }
    else if (camId == CamID::B)
    {
        vector<double> var(6, 0);
        cur_behind_loss_ = opt.cur_behind_loss =
            opt.CostFunction(var, CamID::B, opt.initExt[CamID::B]);
        threads_[0] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -1, 1, -1, 1, -1, 1, -0.005, 0.005,
                             -0.005, 0.005, -0.005, 0.005, camId);
        threads_[1] = thread(&Optimizer::fine_random_search_params, &opt,
                             iter_nums_, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.01,
                             0.01, -0.01, 0.01, -0.01, 0.01, camId);
        threads_[2] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[3] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[4] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
        threads_[5] =
            thread(&Optimizer::fine_random_search_params, &opt, iter_nums_,
                   -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.005, 0.005, -0.005,
                   0.005, -0.005, 0.005, camId);
    }
    for (int i = 0; i < thread_num_; i++)
    {
        threads_[i].join();
    }
    auto end_calib__ = chrono::steady_clock::now();
    double during_calib__ =
        std::chrono::duration<double>(end_calib__ - end_calib_).count();
    cout << "time:" << during_calib__ << endl;
    if (camId == CamID::L)
    {
        // opt.show(CamID::L,prefix+"/after_left_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_left_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_left_loss << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::L] << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[0])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_fl:" << opt.ncoef_fl;
    }
    else if (camId == CamID::B)
    {
        // opt.show(CamID::B,prefix+"/after_behind_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_behind_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_behind_loss
             << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::B] << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[2])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_bl:" << opt.ncoef_bl;
        cout << endl << "ncoef_br:" << opt.ncoef_br;
    }
    else if (camId == CamID::R)
    {
        // opt.show(CamID::R,prefix+"/after_right_calib2.png");
        cout << "luminorsity loss before 2nd opt:" << cur_right_loss_ << endl;
        cout << "luminorsity loss after 2nd opt:" << opt.cur_right_loss << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::R] << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[1])
            cout << fixed << setprecision(6) << e << " ";
        cout << endl << "ncoef_fr:" << opt.ncoef_fr;
    }
    cout << endl;

    cout << "**************************************3rd*************************"
            "***************"
         << endl;
    opt.phase = 3;
    vector<double> var(6, 0);
    double cur_right_loss__, cur_left_loss__, cur_behind_loss__;
    int thread_num__ = 3;
    int iter_nums__  = 20000;
    vector<thread> threads__(thread_num__);
    if (camId == CamID::R)
    {
        vector<double> var(6, 0);
        cur_right_loss__ = opt.cur_right_loss =
            opt.CostFunction(var, CamID::R, opt.optExt[CamID::R]);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, camId);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
    }
    else if (camId == CamID::L)
    {
        vector<double> var(6, 0);
        cur_left_loss__ = opt.cur_left_loss =
            opt.CostFunction(var, CamID::L, opt.optExt[CamID::L]);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, camId);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
    }
    else if (camId == CamID::B)
    {
        vector<double> var(6, 0);
        cur_behind_loss__ = opt.cur_behind_loss =
            opt.CostFunction(var, CamID::B, opt.optExt[CamID::B]);
        threads__[0] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.002, 0.002, -0.002,
                   0.002, -0.002, 0.002, camId);
        threads__[1] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
        threads__[2] =
            thread(&Optimizer::best_random_search_params, &opt, iter_nums__,
                   -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.001, 0.001, -0.001,
                   0.001, -0.001, 0.001, camId);
    }
    for (int i = 0; i < thread_num__; i++)
    {
        threads__[i].join();
    }
    auto end_calib___ = chrono::steady_clock::now();
    double during_calib___ =
        std::chrono::duration<double>(end_calib___ - end_calib__).count();
    cout << "time:" << during_calib___ << endl;
    if (camId == CamID::L)
    {
        // opt.show(CamID::L,prefix+"/after_left_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_left_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_left_loss << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::L] << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.optExt[CamID::L].block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[0])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GL_opt.png",opt.imgl_bev_rgb);
    }
    else if (camId == CamID::B)
    {
        // opt.show(CamID::B,prefix+"/after_behind_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_behind_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_behind_loss
             << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::B] << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.optExt[CamID::B].block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[2])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GB_opt.png",opt.imgb_bev_rgb);
    }
    else if (camId == CamID::R)
    {
        // opt.show(CamID::R,prefix+"/after_right_calib3.png");
        cout << "luminorsity loss before 3rd opt:" << cur_right_loss__ << endl;
        cout << "luminorsity loss after 3rd opt:" << opt.cur_right_loss << endl;
        cout << "extrinsic after opt:" << endl << opt.optExt[CamID::R] << endl;
        cout << "eular:" << endl
             << TransformUtil::Rotation2Eul(
                    opt.optExt[CamID::R].block(0, 0, 3, 3))
             << endl;
        cout << "best search parameters:" << endl;
        for (auto e : opt.bestVal_[1])
            cout << fixed << setprecision(6) << e << " ";
        // imwrite(opt.prefix+"/GR_opt.png",opt.imgr_bev_rgb);
    }
    cout << endl
         << camId << " calibration time: "
         << during_calib_ + during_calib__ + during_calib___ << "s" << endl;
    return during_calib_ + during_calib__ + during_calib___;
}

int main(int argc, char** argv)
{
    if (argc <= 3)
    {
        printf("Usage %s <dataset> <initial calibration> <image set dir> "
               "<output dir>\n",
               argv[0]);
        exit(-1);
    }

    // camera_model:0-fisheye;1-Ocam;2-pinhole
    int camera_model = 0;

    // if add random disturbance to initial pose
    int flag_add_disturbance = 1;

    /*
    solution model :
        1.pure gray pipeline in three phase of optimization :
    solution_model_="gray" 2.(default)Adpative Threshold Binarization in first
    phase and pure gray in the 2nd&3rd phase of optimization:
            solution_model_="atb+gray"
        3.pure Adpative Threshold Binarization in all three phase of
    optimization: solution_model_="atb"
    */
    std::string solution_model_ = "atb+gray";

    // if add road semantic segmentation when in texture extraction process to
    // improve accuracy
    int add_semantic_segmentation_front = 0;
    int add_semantic_segmentation_left  = 0;
    int add_semantic_segmentation_right = 0;

    /*
    read surround frames
    In our demo,imgs1~2 are the surround images captured by a tour
    car(fisheye,Fov=195), imgs3~5 are the surround images captured in Carla
    engine(pinhole,Fov=125) imgs6 are the surround images captured by a car in
    the calibration room(fisheye,Fov=195).
    */

    // bev rows、cols
    int bev_rows = 1000,
        bev_cols = 1000;  // recommendation : pinhole---1500,fisheye---1000

    // if add coarse search(1st search)
    int coarse_search_flag = 1;

    // which camera fixed
    CamID fixed = CamID::F;

    std::string dataset = argv[1];

    calib     = argv[2];
    input     = argv[3];
    output    = argv[4];
    extension = ".png";
    // Mat imgb  = cv::imread(input + "/cam2" + extension);
    // Mat imgf  = cv::imread(input + "/cam1" + extension);
    // Mat imgl  = cv::imread(input + "/cam0" + extension);
    // Mat imgr  = cv::imread(input + "/cam3" + extension);
    Mat imgb = cv::imread(input + "/b" + extension);
    Mat imgf = cv::imread(input + "/f" + extension);
    Mat imgl = cv::imread(input + "/l" + extension);
    Mat imgr = cv::imread(input + "/r" + extension);
    std::filesystem::create_directories(output);

    // topview generator with blending
    using namespace perception::imgproc;
    auto config = std::make_shared<PerceptionConfig>();
    util::LoadProtoFromASCIIFile(
        "/home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/config/"
        "perception_config.textproto",
        config.get());

    // initilize the optimizer
    Optimizer opt(calib, &imgf, &imgl, &imgb, &imgr, camera_model, bev_rows,
                  bev_cols, fixed, coarse_search_flag, dataset,
                  flag_add_disturbance, output, solution_model_);

    // transform extrinsics matrix to be associate with FAPA world frame
    // clang-format off
    Eigen::Matrix4d rotCounterCw90;
    rotCounterCw90 << 0,-1, 0, 0, 
                      1, 0, 0, 0, 
                      0, 0, 1, 0, 
                      0, 0, 0, 1;
    // clang-format on
    // Matrix4d transExtToFapaWorld = rot90.transpose();

    std::array<std::vector<short>, 4> uvLists;
    for (size_t i = 0; i < CamID::NUM_CAM; i++)
    {
        auto camId   = static_cast<CamID>(i);
        Matrix4d ext = opt.initExt[camId] * rotCounterCw90;
        // Matrix4d ext         = opt.initExt[camId];
        Eigen::Matrix3d matR = ext.block<3, 3>(0, 0);
        Eigen::Vector3d vecT = ext.block<3, 1>(0, 3);
        uvLists[i].resize(900 * 800 * 2);
        lut::genLUT(camId, matR, vecT, opt.intrinsics[camId],
                    opt.distortion_params[camId], uvLists[i],
                    output + "/map" + std::to_string((int)camId) + ".txt",
                    false);
    }

    auto imgprocContext = std::make_unique<ImageProcessorContext>(
        std::make_unique<ImageProcessorCuda>(config->imgproc_config()),
        std::make_unique<SegmentImageProcessorCuda>(config->imgproc_config()));

    if (!imgprocContext->init())
    {
        LOG_ERROR("Failed to initialize image processor context!");
        throw std::runtime_error("Cannot init attributes");
    }

    cv::Mat imgTop;
    imgprocContext->createTopViewImage(imgl, imgf, imgb, imgr, imgTop);
    cv::imwrite(output + "/topview_before.png", imgTop);

    // bev images before optimization
    Mat GF = opt.project_on_ground(
        imgf, opt.initExt[CamID::F], opt.intrinsics[CamID::F],
        opt.distortion_params[CamID::F], opt.KG, opt.brows, opt.bcols, opt.hf);
    Mat GB = opt.project_on_ground(
        imgb, opt.initExt[CamID::B], opt.intrinsics[CamID::B],
        opt.distortion_params[CamID::B], opt.KG, opt.brows, opt.bcols, opt.hb);
    Mat GL = opt.project_on_ground(
        imgl, opt.initExt[CamID::L], opt.intrinsics[CamID::L],
        opt.distortion_params[CamID::L], opt.KG, opt.brows, opt.bcols, opt.hl);
    Mat GR = opt.project_on_ground(
        imgr, opt.initExt[CamID::R], opt.intrinsics[CamID::R],
        opt.distortion_params[CamID::R], opt.KG, opt.brows, opt.bcols, opt.hr);

    GF = opt.tail(GF, CamID::F);
    // imwrite(output + "/GF_tail.png", GF);
    // imshow("GF", GF);
    // waitKey(0);
    GB = opt.tail(GB, CamID::B);
    // imwrite(output + "/GB_tail.png", GB);
    // imshow("GB", GB);
    // waitKey(0);

    GL = opt.tail(GL, CamID::L);
    // imwrite(output + "/GL_tail.png", GL);
    // imshow("GL", GL);
    // waitKey(0);

    GR = opt.tail(GR, CamID::R);
    // imwrite(output + "/GR_tail.png", GR);
    // imshow("GR", GR);
    // waitKey(0);

    Mat bev_before = opt.generate_surround_view(GF, GL, GB, GR);
    imwrite(output + "/before_all_calib.png", bev_before);
    // imshow("opt_before", bev_before);
    // waitKey(0);

    // front left field texture extraction
    vector<double> size  = {opt.tailSize[CamID::F], opt.tailSize[CamID::L],
                            opt.tailSize[CamID::B], opt.tailSize[CamID::R]};
    int exposure_flag_fl = 1;  // if add exposure solution
    extractor ext1(GF, GL, add_semantic_segmentation_front, exposure_flag_fl,
                   size);
    if (add_semantic_segmentation_front)
    {
        Mat mask_fl = imread(input + "/mask/front.png");
        ext1.mask_ground.push_back(mask_fl);
    }
    ext1.Binarization();
    ext1.findcontours();
    opt.fl_pixels_texture = ext1.extrac_textures_and_save(
        output + "/texture_fl.png", output + "/fl.csv");
    if (ext1.exposure_flag && ext1.ncoef > 0.5)
    {
        opt.ncoef_fl = ext1.ncoef;
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
    extractor ext2(GF, GR, add_semantic_segmentation_front, exposure_flag_fr,
                   size);
    if (add_semantic_segmentation_front)
    {
        Mat mask_fr = imread(input + "/mask/front.png");
        ext2.mask_ground.push_back(mask_fr);
    }
    ext2.Binarization();
    ext2.findcontours();
    opt.fr_pixels_texture = ext2.extrac_textures_and_save(
        output + "/texture_fr.png", output + "/fr.csv");
    if (ext2.exposure_flag && ext2.ncoef > 0.5)
    {
        opt.ncoef_fr = ext2.ncoef;
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
            "right*************************************"
         << endl;
    double during1 = CameraOptimization(opt, CamID::R);

    cout << "*********************************start "
            "left**************************************"
         << endl;
    double during2 = CameraOptimization(opt, CamID::L);

    // back left field texture extraction
    int exposure_flag_bl = 1;  // if add exposure solution
    cv::imwrite(output + "/GB.png", GB);
    cv::imwrite(output + "/imgl_bev_rgb.png", opt.imgl_bev_rgb);
    extractor ext3(opt.imgl_bev_rgb, GB, add_semantic_segmentation_left,
                   exposure_flag_bl, size);
    if (add_semantic_segmentation_left)
    {
        Mat mask_bl = imread(input + "/mask/left.png");
        ext3.mask_ground.push_back(mask_bl);
    }
    ext3.Binarization();
    ext3.findcontours();
    opt.bl_pixels_texture = ext3.extrac_textures_and_save(
        output + "/texture_bl.png", output + "/bl.csv");
    if (ext3.exposure_flag && ext3.ncoef > 0.5)
    {
        opt.ncoef_bl = ext3.ncoef;
        cout << "ncoef_bl:" << opt.ncoef_bl << endl;
    }
    else
    {
        opt.ncoef_bl = 1;
        cout << "ncoef_bl:" << opt.ncoef_bl << endl;
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
        opt.eigen2mat(opt.KG.inverse()) * pG_bl * opt.hl;
    opt.PG_bl = PG_bl;

    // back right field texture extraction
    int exposure_flag_br = 1;  // if add exposure solution
    cv::imwrite(output + "/GB.png", GB);
    cv::imwrite(output + "/imgr_bev_rgb.png", opt.imgr_bev_rgb);
    extractor ext4(opt.imgr_bev_rgb, GB, add_semantic_segmentation_right,
                   exposure_flag_br, size);
    if (add_semantic_segmentation_right)
    {
        Mat mask_br = imread(input + "/mask/right.png");
        ext4.mask_ground.push_back(mask_br);
    }
    ext4.Binarization();
    ext4.findcontours();
    opt.br_pixels_texture = ext4.extrac_textures_and_save(
        output + "/texture_br.png", output + "/br.csv");
    // opt.br_pixels_texture=opt.readfromcsv(prefix+"/br.csv");
    if (ext4.exposure_flag && ext4.ncoef > 0.5)
    {
        opt.ncoef_br = ext4.ncoef;
        cout << "ncoef_br:" << opt.ncoef_br << endl;
    }
    else
    {
        opt.ncoef_br = 1;
        cout << "ncoef_br:" << opt.ncoef_br << endl;
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
        opt.eigen2mat(opt.KG.inverse()) * pG_br * opt.hr;
    opt.PG_br = PG_br;

    cout << "*********************************start "
            "behind***********************************"
         << endl;
    double during3 = CameraOptimization(opt, CamID::B);

    cout << "************************online calibration "
            "finished!!!**************************"
         << endl;
    cout << "total calibration time:" << during1 + during2 + during3 << "s"
         << endl;

    printf("==================================================================="
           "==\n");
    std::cout << "GT Extrinsics:\n" << opt.gtExt[CamID::B] << "\n";
    std::cout << "Opt Extrinsics:\n" << opt.optExt[CamID::B] << "\n";

    printf("==================================================================="
           "==\n");

    // Open a file for writing
    std::ofstream outputFile(output + "/errors.txt");

    // Check if the file is successfully opened
    if (!outputFile.is_open())
    {
        std::cerr << "Error opening the file for writing." << std::endl;
        return 1;  // Exit with an error code
    }

    outputFile << "CamID - Translation error - Rotation error" << std::endl;

    std::array<std::pair<double, double>, 4> errors;
    errors[CamID::F] =
        util::calculateError(opt.initExt[CamID::F], opt.gtExt[CamID::F]);
    errors[CamID::L] =
        util::calculateError(opt.initExt[CamID::L], opt.gtExt[CamID::L]);
    errors[CamID::B] =
        util::calculateError(opt.initExt[CamID::B], opt.gtExt[CamID::B]);
    errors[CamID::R] =
        util::calculateError(opt.initExt[CamID::R], opt.gtExt[CamID::R]);
    printf("CamID::F - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::F].first, errors[CamID::F].second);
    printf("CamID::L - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::L].first, errors[CamID::L].second);
    printf("CamID::B - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::B].first, errors[CamID::B].second);
    printf("CamID::R - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::R].first, errors[CamID::R].second);

    outputFile << "Before error" << std::endl;
    outputFile << "F: " << errors[CamID::F].first << " "
               << errors[CamID::F].second << std::endl;
    outputFile << "L: " << errors[CamID::L].first << " "
               << errors[CamID::L].second << std::endl;
    outputFile << "B: " << errors[CamID::B].first << " "
               << errors[CamID::B].second << std::endl;
    outputFile << "R: " << errors[CamID::R].first << " "
               << errors[CamID::R].second << std::endl;
    printf("==================================================================="
           "==\n");
    errors[CamID::F] =
        util::calculateError(opt.initExt[CamID::F], opt.gtExt[CamID::F]);
    errors[CamID::L] =
        util::calculateError(opt.optExt[CamID::L], opt.gtExt[CamID::L]);
    errors[CamID::B] =
        util::calculateError(opt.optExt[CamID::B], opt.gtExt[CamID::B]);
    errors[CamID::R] =
        util::calculateError(opt.optExt[CamID::R], opt.gtExt[CamID::R]);
    printf("CamID::F - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::F].first, errors[CamID::F].second);
    printf("CamID::L - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::L].first, errors[CamID::L].second);
    printf("CamID::B - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::B].first, errors[CamID::B].second);
    printf("CamID::R - Translation error: %.5f - Rotation Error: %.5f\n",
           errors[CamID::R].first, errors[CamID::R].second);
    outputFile << "After error" << std::endl;
    outputFile << "F: " << errors[CamID::F].first << " "
               << errors[CamID::F].second << std::endl;
    outputFile << "L: " << errors[CamID::L].first << " "
               << errors[CamID::L].second << std::endl;
    outputFile << "B: " << errors[CamID::B].first << " "
               << errors[CamID::B].second << std::endl;
    outputFile << "R: " << errors[CamID::R].first << " "
               << errors[CamID::R].second << std::endl;
    printf("==================================================================="
           "==\n");

    // Generate LUT
    // std::shared_ptr<std::vector<cv::Mat>>> uvListsPtr;
    // std::vector<cv::Mat> uvLists(4, cv::Mat(900, 800, CV_16SC2));
    for (size_t i = 0; i < CamID::NUM_CAM; i++)
    {
        auto camId   = static_cast<CamID>(i);
        Matrix4d ext = opt.optExt[camId] * rotCounterCw90;
        // Matrix4d ext         = opt.optExt[camId];
        Eigen::Matrix3d matR = ext.block<3, 3>(0, 0);
        Eigen::Vector3d vecT = ext.block<3, 1>(0, 3);
        lut::genLUT(camId, matR, vecT, opt.intrinsics[camId],
                    opt.distortion_params[camId], uvLists[i],
                    output + "/map" + std::to_string((int)camId) + ".txt",
                    false);
    }

    if (!imgprocContext->init())
    {
        LOG_ERROR("Failed to initialize image processor context!");
        throw std::runtime_error("Cannot init attributes");
    }

    imgprocContext->createTopViewImage(imgl, imgf, imgb, imgr, imgTop);

    LOG_INFO("Writing image");
    cv::imwrite(output + "/topview_after.png", imgTop);

    return 0;
}