#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Eigen/Dense>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Optimizer {
private:
  mutable std::mutex mutexleft;
  mutable std::mutex mutexright;
  mutable std::mutex mutexbehind;
  mutable std::mutex mutexval;

public:
  // prefix
  string prefix;

  // camera_model
  int camera_model;

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

  // SVS luminosity loss after optimization
  double cur_left_loss;
  double cur_right_loss;
  double cur_behind_loss;

  // initial SVS luminosity loss
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

  // SVS gray images
  Mat imgf_gray;
  Mat imgl_gray;
  Mat imgb_gray;
  Mat imgr_gray;

  // SVS rgb images
  Mat imgf_rgb;
  Mat imgl_rgb;
  Mat imgb_rgb;
  Mat imgr_rgb;

  // front camera generated BEV image's texture 2d pixels
  vector<Point> fl_pixels_texture;
  vector<Point> fr_pixels_texture;
  vector<Point> bl_pixels_texture;
  vector<Point> br_pixels_texture;

  // front camera generated BEV image's texture 2d pixels-less
  vector<Point> fl_pixels_texture_less;
  vector<Point> fr_pixels_texture_less;
  vector<Point> bl_pixels_texture_less;
  vector<Point> br_pixels_texture_less;

  // euler angles(3) and t parameters(3) to recover precise SVS cameras
  // extrinsics
  vector<vector<double>> bestVal_; //(3*6)

  // SVS generated BEV gray images
  Mat imgf_bev; // initially
  Mat imgl_bev;
  Mat imgr_bev;
  Mat imgb_bev;

  // SVS generated BEV rgb images
  Mat imgf_bev_rgb; // initially
  Mat imgl_bev_rgb;
  Mat imgr_bev_rgb;
  Mat imgb_bev_rgb;

  // Optimizer();
  Optimizer(const Mat *imgf, const Mat *imgl, const Mat *imgb, const Mat *imgr,
            int camera_model_index, int rows, int cols);
  ~Optimizer();
  void initializeK();
  void initializeD();
  void initializePose();
  void initializeKG();
  void initializeHeight();
  Mat tail(Mat img, string index);
  void Calibrate_left(int search_count, double roll_ep0, double roll_ep1,
                      double pitch_ep0, double pitch_ep1, double yaw_ep0,
                      double yaw_ep1, double t0_ep0, double t0_ep1,
                      double t1_ep0, double t1_ep1, double t2_ep0,
                      double t2_ep1);
  void Calibrate_right(int search_count, double roll_ep0, double roll_ep1,
                       double pitch_ep0, double pitch_ep1, double yaw_ep0,
                       double yaw_ep1, double t0_ep0, double t0_ep1,
                       double t1_ep0, double t1_ep1, double t2_ep0,
                       double t2_ep1);
  void Calibrate_behind(int search_count, double roll_ep0, double roll_ep1,
                        double pitch_ep0, double pitch_ep1, double yaw_ep0,
                        double yaw_ep1, double t0_ep0, double t0_ep1,
                        double t1_ep0, double t1_ep1, double t2_ep0,
                        double t2_ep1);
  void fine_Calibrate_right(int search_count, double roll_ep0, double roll_ep1,
                            double pitch_ep0, double pitch_ep1, double yaw_ep0,
                            double yaw_ep1, double t0_ep0, double t0_ep1,
                            double t1_ep0, double t1_ep1, double t2_ep0,
                            double t2_ep1);
  void fine_Calibrate_left(int search_count, double roll_ep0, double roll_ep1,
                           double pitch_ep0, double pitch_ep1, double yaw_ep0,
                           double yaw_ep1, double t0_ep0, double t0_ep1,
                           double t1_ep0, double t1_ep1, double t2_ep0,
                           double t2_ep1);
  void fine_Calibrate_behind(int search_count, double roll_ep0, double roll_ep1,
                             double pitch_ep0, double pitch_ep1, double yaw_ep0,
                             double yaw_ep1, double t0_ep0, double t0_ep1,
                             double t1_ep0, double t1_ep1, double t2_ep0,
                             double t2_ep1);
  double CostFunction(const vector<double> var, string idx, string model);
  double fine_CostFunction(const vector<double> var, string idx, string model);
  void SaveOptResult(const string img_name);
  void show(string idx, string filename);
  cv::Mat eigen2mat(Eigen::MatrixXd A);
  Mat gray_gamma(Mat img);
  void world2cam(double point2D[2], double point3D[3], Eigen::Matrix3d K,
                 vector<double> D);
  void distortPoints(cv::Mat &P_GC1, cv::Mat &p_GC, Eigen::Matrix3d &K_C);
  void distortPointsOcam(cv::Mat &P_GC1, cv::Mat &p_GC, Eigen::Matrix3d &K_C,
                         vector<double> &D_C);
  void random_search_params(int search_count, double roll_ep0, double roll_ep1,
                            double pitch_ep0, double pitch_ep1, double yaw_ep0,
                            double yaw_ep1, double t0_ep0, double t0_ep1,
                            double t1_ep0, double t1_ep1, double t2_ep0,
                            double t2_ep1, string idx, string model);
  void fine_random_search_params(int search_count, double roll_ep0,
                                 double roll_ep1, double pitch_ep0,
                                 double pitch_ep1, double yaw_ep0,
                                 double yaw_ep1, double t0_ep0, double t0_ep1,
                                 double t1_ep0, double t1_ep1, double t2_ep0,
                                 double t2_ep1, string idx, string model);
  // double extract_commonview_and_compute_loss(Mat img1,Mat img2,string
  // idx,string model);
  Mat generate_surround_view(Mat img_GF, Mat img_GL, Mat img_GB, Mat img_GR);
  vector<Point> readfromcsv(string filename);
  Mat project_on_ground(cv::Mat img, Eigen::Matrix4d T_CG, Eigen::Matrix3d K_C,
                        vector<double> D_C, Eigen::Matrix3d K_G, int rows,
                        int cols, float height);
  double back_camera_and_compute_loss(Mat img1, Mat img2, Eigen::Matrix4d T,
                                      string idx, string model, double height);
  double getPixelValue(Mat *image, float x, float y);
  // Mat project_on_ground(Mat img, Eigen::Matrix4d T);
  // void distortPoints(Mat &P_GC1,Mat &p_GC);
  // Mat eigen2mat(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A);
  // vector<vector<Point>> fillContour(const vector<vector<Point>> contours);
};

#endif //  OPTIMIZER_H_