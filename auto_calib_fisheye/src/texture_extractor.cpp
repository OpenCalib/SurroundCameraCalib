#include "texture_extractor.h"
#include <algorithm>
#include <fstream>
extractor::extractor(Mat img1_bev_, Mat img2_bev_, int edge_flag_,
                     int exposure_flag_, vector<double> size)
{
    img1_bev = img1_bev_;
    img2_bev = img2_bev_;

    edge_flag     = edge_flag_;
    exposure_flag = exposure_flag_;

    Mat bev;
    addWeighted(img1_bev, 0.5, img2_bev, 0.5, 0, bev);
    bev_of_imgs = bev;

    sizef = size[0];
    sizel = size[1];
    sizeb = size[2];
    sizer = size[3];
}

extractor::~extractor() {}

void extractor::writetocsv(string filename, vector<Point> vec)
{
    ofstream outfile;
    outfile.open(filename, ios::out);
    for (auto pixel : vec)
    {
        outfile << pixel.x << " " << pixel.y << endl;
    }
    outfile.close();
}

void extractor::Binarization()
{
    assert(img1_bev.rows == img2_bev.rows);
    assert(img1_bev.cols == img2_bev.cols);
    int rows = img1_bev.rows;
    int cols = img2_bev.cols;
    Mat dst(img1_bev.size(), CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (img1_bev.at<Vec3b>(i, j) != Vec3b(0, 0, 0) &&
                img2_bev.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
            {
                dst.at<uchar>(i, j) = 255;
            }
            else
            {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
    bin_of_imgs = dst;

    // imshow("bin",bin_of_imgs);
    // waitKey(0);
}

void extractor::findcontours()
{
    vector<vector<Point>> contours;
    Mat erode_img, dilate_img;
    // Mat dilate_kernel = getStructuringElement(0, Size(5,5));
    // dilate(bin_of_imgs,dilate_img,dilate_kernel);
    Mat erode_kernel = getStructuringElement(0, Size(7, 7));
    erode(bin_of_imgs, erode_img, erode_kernel, Point(-1, -1), 2);
    cv::findContours(erode_img, contours, cv::noArray(), CV_RETR_EXTERNAL,
                     CV_CHAIN_APPROX_NONE);

    int maxsize = 50;
    int index   = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > maxsize)
        {
            maxsize = contours[i].size();
            index   = i;
        }
    }
    vector<vector<Point>> contours_after_filter;
    printf("contours.size() = %lu\n", contours.size());
    contours_after_filter.push_back(contours[index]);
    ;
    std::vector<std::vector<cv::Point>> contours_pixels;
    contours_pixels = fillContour(contours_after_filter);

    this->contours = contours_pixels;

    // for(auto e:contours_pixels[0]){
    // 	cv::circle(bev_of_imgs,e,0,Scalar(0,255,0),-1);
    // }
    // imshow("contours",bev_of_imgs);
    // waitKey(0);
}

std::vector<std::vector<cv::Point>> extractor::fillContour(
    const std::vector<std::vector<cv::Point>>& _contours)
{
    // sort as x descent y descent.
    std::vector<std::vector<cv::Point>> contours(_contours);
    for (size_t i = 0; i < contours.size(); ++i)
    {
        std::vector<cv::Point> sub(contours[i]);
        std::sort(sub.begin(), sub.end(), [&](cv::Point& A, cv::Point& B) {
            if (A.x == B.x)
                return A.y < B.y;
            else
                return A.x < B.x;
        });
        contours[i] = sub;
    }
    // restore as pairs with same ys.
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours_pair;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        std::vector<std::pair<cv::Point, cv::Point>> pairs;
        for (size_t j = 0; j < contours[i].size(); ++j)
        {
            // j==0
            if (pairs.size() == 0)
            {
                pairs.push_back({contours[i][j], contours[i][j]});
                continue;
            }
            // j>0
            if (contours[i][j].x != pairs[pairs.size() - 1].first.x)
            {
                pairs.push_back({contours[i][j], contours[i][j]});
                continue;
            }

            if (contours[i][j].x == pairs[pairs.size() - 1].first.x)
            {
                if (contours[i][j].y > pairs[pairs.size() - 1].second.y)
                    pairs[pairs.size() - 1].second = contours[i][j];
                continue;
            }
        }
        contours_pair.push_back(pairs);
    }

    // fill contour coordinates.
    std::vector<std::vector<cv::Point>> fill_con;
    for (auto pair_set : contours_pair)
    {
        std::vector<cv::Point> pointSet;
        for (auto aPair : pair_set)
        {
            if (aPair.first == aPair.second)
            {
                pointSet.push_back(aPair.first);
            }
            else
            {
                for (int i = aPair.first.y; i <= aPair.second.y; ++i)
                {
                    pointSet.push_back(cv::Point(aPair.first.x, i));
                }
            }
        }
        fill_con.push_back(pointSet);
    }
    return fill_con;
}

vector<pair<cv::Point, double>> extractor::extrac_textures_and_save(
    string pic_filename, string csv_filename)
{
    int down_sample = 800;
    Mat gray1, gray2;
    cvtColor(img1_bev, gray1, COLOR_BGR2GRAY);
    cvtColor(img2_bev, gray2, COLOR_BGR2GRAY);
    // adaptiveThreshold(gray1, gray1, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
    // THRESH_BINARY, 17, -3); adaptiveThreshold(gray2, gray2, 255,
    // ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 17, -3);
    if (exposure_flag)
    {
        Mat ROI1, ROI2;
        bitwise_and(gray1, bin_of_imgs, ROI1);
        // imshow("ROI1",ROI1);
        // waitKey(0);
        bitwise_and(gray2, bin_of_imgs, ROI2);
        // imshow("ROI2",ROI2);
        // waitKey(0);

        double mean1 = mean(ROI1).val[0];
        double mean2 = mean(ROI2).val[0];
        ncoef        = mean1 / mean2;

        // Scalar mean1,mean2;
        // Scalar std1,std2;
        // meanStdDev(ROI1,mean1,std1);
        // meanStdDev(ROI2,mean2,std2);
    }
    GaussianBlur(gray1, gray1, Size(3, 3), 0);
    GaussianBlur(gray2, gray2, Size(3, 3), 0);

    vector<pair<cv::Point, double>> texture;
    vector<Point> contour1_pixels = contours[0];
    double max_norm               = INT_MIN;
    double min_norm               = INT_MAX;
    for (auto pixel : contour1_pixels)
    {
        int flag_ground = 0;

        if (edge_flag)
        {
            // mask-filter
            for (auto mask : mask_ground)
            {
                if (mask.at<Vec3b>(pixel.y, pixel.x) == Vec3b(255, 255, 255))
                    flag_ground = 1;
            }
        }

        if (flag_ground || !edge_flag)
        {
            if (pixel.x < 10 || pixel.y < 10 ||
                (pixel.x + 10) > img1_bev.cols ||
                (pixel.y + 10) > img1_bev.rows)
                continue;

            Eigen::Vector2d delta((gray1.ptr<uchar>(pixel.y)[pixel.x] -
                                   gray1.ptr<uchar>(pixel.y)[pixel.x - 2]),
                                  (gray1.ptr<uchar>(pixel.y)[pixel.x] -
                                   gray1.ptr<uchar>(pixel.y - 2)[pixel.x]));
            if (delta.norm() < 15) continue;
            pair<cv::Point, double> pixel_g(pixel, delta.norm());
            texture.push_back(pixel_g);
        }
    }
    vector<pair<cv::Point, double>> texture_down;
    Mat show = img1_bev.clone();

    // adaptiveThreshold(show, show, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
    // THRESH_BINARY, 15, -2); GaussianBlur(show,show,Size(5,5),0);
    // cvtColor(show,show,CV_GRAY2BGR);

    // 1:random down sample (num=down_sample)
    //  srand((int)time(0));
    //  for(int i=0;i<down_sample;i++){
    //      int k=rand()%texture.size();
    //      texture_down.push_back(texture[k]);
    //      circle(show, texture[k].first, 1,  Scalar(255, 0, 0), -1);
    //  }

    // 2:do not down sample but filte local pixles
    //  for(int i=0;i<texture.size();i++){
    //  	if(local_pixel_test(texture_down,texture[i]))
    //  		continue;
    //  	max_norm=max(max_norm,texture[i].second);
    //  	min_norm=min(min_norm,texture[i].second);
    //  	texture[i].second=max_norm/texture[i].second;
    //  	texture_down.push_back(texture[i]);
    //  	circle(show, texture[i].first, 1,  Scalar(0, 255, 0), -1);
    //  }

    // 3:random down sample(num=down_sample) and filte local pixels
    int index = 0;
    srand((int)time(0));
    for (int i = 0; i < texture.size(); i++)
    {
        if (index == down_sample) break;
        int k = rand() % texture.size();
        if (local_pixel_test(texture_down, texture[k])) continue;
        max_norm          = max(max_norm, texture[k].second);
        min_norm          = min(min_norm, texture[k].second);
        texture[k].second = max_norm / texture[k].second;
        texture_down.push_back(texture[k]);
        circle(show, texture[k].first, 1, Scalar(0, 255, 0), -1);
        index++;
    }

    // cout<<"max_norm:"<<max_norm<<endl;
    // cout<<"min_norm:"<<min_norm<<endl;
    // imwrite(pic_filename,show);
    // writetocsv(csv_filename,texture_down);
    return texture_down;
}

// judge if there are pixels in 3*3 local area
bool extractor::local_pixel_test(vector<pair<cv::Point, double>> texture,
                                 pair<cv::Point, double> pixel)
{
    vector<cv::Point> texture_;
    for (auto e : texture)
    {
        texture_.push_back(e.first);
    }
    int x = pixel.first.x;
    int y = pixel.first.y;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (find(texture_.begin(), texture_.end(), Point(x + i, y + j)) !=
                texture_.end())
            {
                return true;
            }
        }
    }
    return false;
}
