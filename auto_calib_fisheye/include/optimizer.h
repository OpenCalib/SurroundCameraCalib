#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Dense>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Optimizer
{
private:
    mutable std::mutex mutexfront;
    mutable std::mutex mutexleft;
    mutable std::mutex mutexright;
    mutable std::mutex mutexbehind;
    mutable std::mutex mutexval;

public:
    // path to load initialize extrinsics
    std::string calibPath_;

    // optimization phase(1/2/3)
    int phase;

    // solution model
    string solution_model;

    // data set index
    string data_index;

    // prefix
    string prefix;

    // camera_model
    int camera_model;

    // which camera fixed
    string fixed;

    // if add coarse search(1st search)
    int coarse_flag;

    // if add disturbance
    int flag_add_disturbance;

    // bev rows、cols
    int brows;
    int bcols;

    // SVS cameras intrinsic-T
    Eigen::Matrix3d intrinsic_front;
    Eigen::Matrix3d intrinsic_left;
    Eigen::Matrix3d intrinsic_behind;
    Eigen::Matrix3d intrinsic_right;

    // K_G
    Eigen::Matrix3d KG;

    // initial SVS cameras extrinsics-T(SVS cameras->BEV)
    Eigen::Matrix4d extrinsic_front;
    Eigen::Matrix4d extrinsic_left;
    Eigen::Matrix4d extrinsic_behind;
    Eigen::Matrix4d extrinsic_right;

    // SVS cameras extrinsics-T after optimization
    Eigen::Matrix4d extrinsic_front_opt;
    Eigen::Matrix4d extrinsic_left_opt;
    Eigen::Matrix4d extrinsic_behind_opt;
    Eigen::Matrix4d extrinsic_right_opt;

    // distortion papameters
    vector<double> distortion_params_front;
    vector<double> distortion_params_left;
    vector<double> distortion_params_behind;
    vector<double> distortion_params_right;

    // SVS cameras height
    double hf, hl, hb, hr;

    // tail size
    double sizef, sizel, sizeb, sizer;

    // SVS luminosity loss after optimization
    double cur_front_loss;
    double cur_left_loss;
    double cur_right_loss;
    double cur_behind_loss;

    // initial SVS luminosity loss
    double max_front_loss;
    double max_left_loss;
    double max_right_loss;
    double max_behind_loss;

    // bev texture pixel
    Mat pG_fl;
    Mat pG_fr;
    Mat pG_bl;
    Mat pG_br;

    // bev texture project to camera coordinate
    Mat PG_fl;
    Mat PG_fr;
    Mat PG_bl;
    Mat PG_br;

    // SVS gray images(gamma)
    Mat imgf_gray;
    Mat imgl_gray;
    Mat imgb_gray;
    Mat imgr_gray;
    // SVS gray images(adaptive threshold binary)
    Mat imgf_atb;
    Mat imgl_atb;
    Mat imgb_atb;
    Mat imgr_atb;

    // SVS rgb images
    Mat imgf_rgb;
    Mat imgl_rgb;
    Mat imgb_rgb;
    Mat imgr_rgb;

    // front camera generated BEV image's texture 2d pixels
    vector<pair<Point, double>> fl_pixels_texture;
    vector<pair<Point, double>> fr_pixels_texture;
    vector<pair<Point, double>> bl_pixels_texture;
    vector<pair<Point, double>> br_pixels_texture;

    // euler angles(3) and t parameters(3) to recover precise SVS cameras
    // extrinsics
    vector<vector<double>> bestVal_;  //(3*6)

    // SVS generated BEV gray images
    Mat imgf_bev_gray;  // initially
    Mat imgl_bev_gray;
    Mat imgr_bev_gray;
    Mat imgb_bev_gray;

    // SVS generated BEV gray images
    Mat imgf_bev_atb;  // initially
    Mat imgl_bev_atb;
    Mat imgr_bev_atb;
    Mat imgb_bev_atb;

    // SVS generated BEV rgb images
    Mat imgf_bev_rgb;  // initially
    Mat imgl_bev_rgb;
    Mat imgr_bev_rgb;
    Mat imgb_bev_rgb;

    // ncoef-commonView mean luminorsity ratio
    double ncoef_fl, ncoef_fr, ncoef_bl, ncoef_br;

    // Optimizer();
    Optimizer(const std::string& calibPath, const Mat *imgf, const Mat *imgl, const Mat *imgb,
              const Mat *imgr, int camera_model_index, int rows, int cols,
              string first_order, int flag, string data_set,
              int flag_add_disturbance, string prefix, string solution_model_);
    ~Optimizer();
    void initializeK();
    void initializeD();
    void initializePose();
    void initializeKG();
    void initializeHeight();
    void initializetailsize();
    Mat tail(Mat img, string index);
    double CostFunction(const vector<double> var, string idx,
                        Eigen::Matrix4d T);
    void SaveOptResult(const string img_name);
    void show(string idx, string filename);
    Mat eigen2mat(Eigen::MatrixXd A);
    Mat gray_gamma(Mat img);
    Mat gray_atb(Mat img);
    void world2cam(double point2D[2], double point3D[3], Eigen::Matrix3d K,
                   vector<double> D);
    void distortPoints(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C);
    void distortPointsOcam(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C,
                           vector<double> &D_C);
    void random_search_params(int search_count, double roll_ep0,
                              double roll_ep1, double pitch_ep0,
                              double pitch_ep1, double yaw_ep0, double yaw_ep1,
                              double t0_ep0, double t0_ep1, double t1_ep0,
                              double t1_ep1, double t2_ep0, double t2_ep1,
                              string idx);
    void fine_random_search_params(int search_count, double roll_ep0,
                                   double roll_ep1, double pitch_ep0,
                                   double pitch_ep1, double yaw_ep0,
                                   double yaw_ep1, double t0_ep0, double t0_ep1,
                                   double t1_ep0, double t1_ep1, double t2_ep0,
                                   double t2_ep1, string idx);
    void best_random_search_params(int search_count, double roll_ep0,
                                   double roll_ep1, double pitch_ep0,
                                   double pitch_ep1, double yaw_ep0,
                                   double yaw_ep1, double t0_ep0, double t0_ep1,
                                   double t1_ep0, double t1_ep1, double t2_ep0,
                                   double t2_ep1, string idx);
    Mat generate_surround_view(Mat img_GF, Mat img_GL, Mat img_GB, Mat img_GR);
    Mat generate_surround_viewX(Mat img_GF, Mat img_GL, Mat img_GB, Mat img_GR);
    vector<Point> readfromcsv(string filename);
    Mat Binarization(Mat img1, Mat img2);
    Mat project_on_ground(Mat img, Eigen::Matrix4d T_CG, Eigen::Matrix3d K_C,
                          vector<double> D_C, Eigen::Matrix3d K_G, int rows,
                          int cols, float height);
    double back_camera_and_compute_loss(Mat img1_bev_gray, Mat img1_bev_atb,
                                        Mat img2_gray, Mat img2_atb,
                                        Eigen::Matrix4d T, string idx);
    double getPixelValue(Mat *image, float x, float y);
};

#endif  //  OPTIMIZER_H_