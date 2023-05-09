#include<opencv2/opencv.hpp>
#include<iostream>
#include<opencv2/features2d/features2d.hpp>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<glog/logging.h>
#include<opencv2/core/eigen.hpp>
#include<stdlib.h> 
#include<time.h> 
#include<math.h>
#include<random>
using namespace std;
using namespace cv;
using namespace Eigen;

class extractor{
public:

    Mat img1_bev;
    Mat img2_bev;

    Mat intrinsic1;
    Mat intrinsic2;

    Mat bin_of_imgs;

    Mat bev_of_imgs;

    vector<vector<Point>> contours;

    Eigen::Matrix4d extrinsic1;
    Eigen::Matrix4d extrinsic2;     

    extractor(Mat img1_bev,Mat img2_bev);
    extractor(Mat img1_bev,Mat img2_bev,
              Mat intrinsic1,Mat intrinsic2,
              Eigen::Matrix4d extrinsic1,Eigen::Matrix4d extrinsic2);
    ~extractor();
    void Binarization();
    void writetocsv(string filename,vector<Point>vec);
    void findcontours();
    std::vector<std::vector<cv::Point>> fillContour(const std::vector<std::vector<cv::Point>> & _contours);
    Mat extrac_textures_and_save(string filename);
    vector<cv::Point> extrac_textures();
    static bool cmpx(Point p1, Point p2){ 
        return p1.x < p2.x;
    };
    static bool cmpy(Point p1, Point p2){ 
        return p1.y < p2.y;
    };
};