#include <glog/logging.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
using namespace std;
using namespace cv;
using namespace Eigen;

class extractor
{
public:
    Mat img1_bev;
    Mat img2_bev;

    Mat bin_of_imgs;  // binary common-view bev img
    Mat bev_of_imgs;  // rgb common-view bev img

    vector<Mat> mask_ground;

    int edge_flag;  // if filter edge

    int exposure_flag;  // if add exposure solution

    double ncoef;  // exposure coefficients

    vector<vector<Point>> contours;  // common-view pixels

    double sizef, sizel, sizeb, sizer;  // bev size of surround-cameras

    extractor(Mat img1_bev, Mat img2_bev, int edge_flag, int exposure_flag,
              vector<double> size);
    ~extractor();
    void Binarization();
    void writetocsv(string filename, vector<Point> vec);
    void findcontours();
    bool local_pixel_test(vector<pair<cv::Point, double>> texture,
                          pair<cv::Point, double> pixel);
    std::vector<std::vector<cv::Point>> fillContour(
        const std::vector<std::vector<cv::Point>>& _contours);
    vector<pair<cv::Point, double>> extrac_textures_and_save(
        string pic_filename, string csv_filename);
    static bool cmpx(Point p1, Point p2) { return p1.x < p2.x; };
    static bool cmpy(Point p1, Point p2) { return p1.y < p2.y; };
};