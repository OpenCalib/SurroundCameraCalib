#include "optimizer.h"
#include <fstream>
#include <random>
#include "transform_util.h"

Mat Optimizer::eigen2mat(Eigen::MatrixXd A)
{
    Mat B;
    eigen2cv(A, B);

    return B;
}

Mat Optimizer::gray_gamma(Mat img)
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // gamma变换
    double contrast   = 1.1;
    double brightness = 0;
    double delta      = 30;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int g                = gray.at<uchar>(i, j);
            gray.at<uchar>(i, j) = saturate_cast<uchar>(
                contrast * (gray.at<uchar>(i, j) - delta) + brightness);
        }
    }

    // imshow("gray",gray);
    // waitKey(0);

    return gray;
}

Mat Optimizer::gray_atb(Mat img)
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // 自适应二值化
    adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
                      17, -3);

    // imshow("gray",gray);
    // waitKey(0);
    return gray;
}

double Optimizer::getPixelValue(Mat *image, float x, float y)
{
    // 法1：双线性插值
    uchar *data = &image->data[int(y) * image->step + int(x)];
    float xx    = x - floor(x);
    float yy    = y - floor(y);
    return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
                 (1 - xx) * yy * data[image->step] +
                 xx * yy * data[image->step + 1]);
}

vector<Point> Optimizer::readfromcsv(string filename)
{
    vector<Point> pixels;
    ifstream inFile(filename, ios::in);
    string lineStr;
    while (getline(inFile, lineStr))
    {
        Point pixel;
        istringstream record(lineStr);
        string x, y;
        record >> x;
        pixel.x = atoi(x.c_str());
        record >> y;
        pixel.y = atoi(y.c_str());
        pixels.push_back(pixel);
    }
    return pixels;
}

Mat Optimizer::Binarization(Mat img1, Mat img2)
{
    assert(img1.rows == img2.rows);
    assert(img1.cols == img2.cols);
    int rows = img1.rows;
    int cols = img2.cols;
    Mat dst(img1.size(), CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (img1.at<Vec3b>(i, j) != Vec3b(0, 0, 0) &&
                img2.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
            {
                dst.at<uchar>(i, j) = 255;
            }
            else
            {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
    return dst;
    // imshow("bin",dst);
    // waitKey(0);
}

Mat Optimizer::tail(Mat img, string index)
{
    if (index == "f")
    {
        Rect m_select_f     = Rect(0, 0, img.cols, sizef);
        Mat cropped_image_f = img(m_select_f);
        Mat border(img.rows - sizef, img.cols, cropped_image_f.type(),
                   Scalar(0, 0, 0));
        Mat dst_front;
        vconcat(cropped_image_f, border, dst_front);
        return dst_front;
    }
    else if (index == "l")
    {
        Rect m_select_l     = Rect(0, 0, sizel, img.rows);
        Mat cropped_image_l = img(m_select_l);
        Mat border2(img.rows, img.cols - sizel, cropped_image_l.type(),
                    Scalar(0, 0, 0));
        Mat dst_left;
        hconcat(cropped_image_l, border2, dst_left);
        return dst_left;
    }
    else if (index == "b")
    {
        Rect m_select_b     = Rect(0, img.rows - sizeb, img.cols, sizeb);
        Mat cropped_image_b = img(m_select_b);
        Mat border1(img.rows - sizeb, img.cols, cropped_image_b.type(),
                    Scalar(0, 0, 0));
        Mat dst_behind;
        vconcat(border1, cropped_image_b, dst_behind);
        return dst_behind;
    }
    else if (index == "r")
    {
        Rect m_select_r     = Rect(img.cols - sizer, 0, sizer, img.rows);
        Mat cropped_image_r = img(m_select_r);
        Mat border3(img.rows, img.cols - sizer, cropped_image_r.type(),
                    Scalar(0, 0, 0));
        Mat dst_right;
        hconcat(border3, cropped_image_r, dst_right);
        return dst_right;
    }
    return Mat(img.rows, img.cols, img.type());
}

void Optimizer::SaveOptResult(const string filename)
{
    // do not tail imgs
    // Mat
    // imgf_bev_rgb_=project_on_ground(imgf_rgb,extrinsic_front_opt,intrinsic_front,distortion_params_front,KG,brows,bcols,hb);
    // Mat
    // imgl_bev_rgb_=project_on_ground(imgl_rgb,extrinsic_left_opt,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
    // Mat
    // imgr_bev_rgb_=project_on_ground(imgr_rgb,extrinsic_right_opt,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
    // Mat
    // imgb_bev_rgb_=project_on_ground(imgb_rgb,extrinsic_behind_opt,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hb);

    //
    Mat opt_after  = generate_surround_view(imgf_bev_rgb, imgl_bev_rgb,
                                            imgb_bev_rgb, imgr_bev_rgb);
    Mat opt_after1 = generate_surround_viewX(imgf_bev_rgb, imgl_bev_rgb,
                                             imgb_bev_rgb, imgr_bev_rgb);
    imwrite(filename + ".png", opt_after);
    imwrite(filename + "X.png", opt_after1);
    // imshow("after_all_cameras_calib",opt_after);
    // waitKey(0);
}

void Optimizer::show(string idx, string filename)
{
    Mat dst, dst1;

    if (idx == "right")
    {  // first
        if (fixed == "front")
            addWeighted(imgf_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
        else
            addWeighted(imgb_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
        // imshow("after_right_calib",dst);
        // waitKey(0);
        imwrite(filename, dst);
    }
    if (idx == "behind")
    {  // second
        addWeighted(imgb_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
        addWeighted(dst, 1, imgl_bev_rgb, 0.5, 3, dst1);
        // imshow("after_behind_calib",dst1);
        // waitKey(0);
        imwrite(filename, dst1);
    }
    if (idx == "left")
    {  // third
        if (fixed == "front")
            addWeighted(imgl_bev_rgb, 0.5, imgf_bev_rgb, 0.5, 3, dst1);
        else
            addWeighted(imgl_bev_rgb, 0.5, imgb_bev_rgb, 0.5, 3, dst1);
        // imshow("after_left_calib",dst1);
        // waitKey(0);
        imwrite(filename, dst1);
    }
    if (idx == "front")
    {
        addWeighted(imgf_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
        addWeighted(dst, 1, imgl_bev_rgb, 0.5, 3, dst1);
        // imshow("after_front_calib",dst1);
        // waitKey(0);
        imwrite(filename, dst1);
    }
}

void Optimizer::world2cam(double point2D[2], double point3D[3],
                          Eigen::Matrix3d K, vector<double> D)
{
    double norm  = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
    double theta = atan(point3D[2] / norm);
    double t, t_i;
    double rho, x, y;
    double invnorm;
    int i;

    if (norm != 0)
    {
        invnorm = 1 / norm;
        t       = theta;
        rho     = D[0];
        t_i     = 1;

        for (i = 1; i < D.size(); i++)
        {
            t_i *= t;
            rho += t_i * D[i];
        }

        x = point3D[0] * invnorm * rho;
        y = point3D[1] * invnorm * rho;

        point2D[0] = x * K(0, 0) + y * K(0, 1) + K(0, 2);
        point2D[1] = x * K(1, 0) + y + K(1, 2);
    }
    else
    {
        point2D[0] = K(0, 2);
        point2D[1] = K(1, 2);
    }
}

void Optimizer::distortPointsOcam(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C,
                                  vector<double> &D_C)
{
    double M[3];
    double m[2];
    for (int i = 0; i < P_GC1.cols; i++)
    {
        M[0] = P_GC1.at<Vec2d>(0, i)[0];
        M[1] = P_GC1.at<Vec2d>(0, i)[1];
        M[2] = -1;
        world2cam(m, M, K_C, D_C);
        p_GC.at<Vec2d>(0, i)[0] = m[0];
        p_GC.at<Vec2d>(0, i)[1] = m[1];
    }
}

void Optimizer::distortPoints(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C)
{
    for (int i = 0; i < P_GC1.cols; i++)
    {
        double x = P_GC1.at<Vec2d>(0, i)[0];
        double y = P_GC1.at<Vec2d>(0, i)[1];

        double u = x * K_C(0, 0) + K_C(0, 2);
        double v = y * K_C(1, 1) + K_C(1, 2);

        p_GC.at<Vec2d>(0, i)[0] = u;
        p_GC.at<Vec2d>(0, i)[1] = v;
    }
}

void Optimizer::initializeK()
{
    Eigen::Matrix3d K_F;
    Eigen::Matrix3d K_L;
    Eigen::Matrix3d K_B;
    Eigen::Matrix3d K_R;

    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        K_F << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750,
            0.00000000, 0.00000000, 1.00000000;
        K_L << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750,
            0.00000000, 0.00000000, 1.00000000;
        K_B << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750,
            0.00000000, 0.00000000, 1.00000000;
        K_R << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750,
            0.00000000, 0.00000000, 1.00000000;
    }

    if (data_index == "imgs1" || data_index == "imgs2")
    {
        K_L << 304.007121, 0.0, 638.469054, 0.0, 304.078429, 399.956311, 0.0,
            0.0, 1.0;
        K_F << 304.007121, 0.0, 638.469054, 0.0, 304.078429, 399.956311, 0.0,
            0.0, 1.0;
        K_B << 304.007121, 0.0, 638.469054, 0.0, 304.078429, 399.956311, 0.0,
            0.0, 1.0;
        K_R << 304.007121, 0.0, 638.469054, 0.0, 304.078429, 399.956311, 0.0,
            0.0, 1.0;
        printf("Set custom K\n");

        // K_F << 422.13163849, 0.00000000, 612.82890504, 0.00000000,
        // 421.10340889,
        //     545.05656249, 0.00000000, 0.00000000, 1.00000000;
        // K_L << 420.60079305, 0.00000000, 650.54173853, 0.00000000,
        // 418.94827188,
        //     527.27178143, 0.00000000, 0.00000000, 1.00000000;
        // K_B << 422.61569350, 0.00000000, 632.46019501, 0.00000000,
        // 421.27373079,
        //     548.34673288, 0.00000000, 0.00000000, 1.00000000;
        // K_R << 421.64203585, 0.00000000, 640.09362064, 0.00000000,
        // 420.26647020,
        //     529.05566315, 0.00000000, 0.00000000, 1.00000000;

        // [[304.007121, 0.0, 638.469054], [0.0, 304.078429, 399.956311], [0.0,
        // 0.0, 1.0]]
    }

    if (data_index == "imgs6")
    {
        K_F << 447.2320251464844, 0, 943.4813232421875, 0, 447.03472900390625,
            757.8351440429688, 0, 0, 1;
        K_L << 451.9393005371094, 0, 966.8802490234375, 0, 452.0724182128906,
            770.867919921875, 0, 0, 1;
        K_B << 449.09722900390625, 0, 959.5527954101563, 0, 449.27923583984375,
            764.47607421875, 0, 0, 1;
        K_R << 451.8409118652344, 0, 962.7486572265625, 0, 451.91888427734375,
            765.1062622070313, 0, 0, 1;
    }

    intrinsic_front  = K_F;
    intrinsic_left   = K_L;
    intrinsic_behind = K_B;
    intrinsic_right  = K_R;
    return;
}

void Optimizer::initializeD()
{
    vector<double> D_F;
    vector<double> D_L;
    vector<double> D_B;
    vector<double> D_R;

    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        D_F = {0, 0, 0, 0};
        D_L = {0, 0, 0, 0};
        D_B = {0, 0, 0, 0};
        D_R = {0, 0, 0, 0};
    }

// {"mean": 0.728127, "max": 1.592387, "camera_matrix": [[304.007121, 0.0, 638.469054], [0.0, 304.078429, 399.956311], [0.0, 0.0, 1.0]], "dist_coefs": [[0.138281], [0.025172], [-0.030963], [0.005019]]}

    if (data_index == "imgs1" || data_index == "imgs2")
    {
        printf("Set custom D\n");
        D_L = {0.138281, 0.025172, -0.030963, 0.005019};
        D_F = {0.138281, 0.025172, -0.030963, 0.005019};
        D_B = {0.138281, 0.025172, -0.030963, 0.005019};
        D_R = {0.138281, 0.025172, -0.030963, 0.005019};
    }

    if (data_index == "imgs6")
    {
        D_F = {0.10423154383897781, -0.058837272226810455, 0.044785112142562866,
               -0.011534594930708408};
        D_L = {
            0.058297254145145416,
            0.016390923410654068,
            -0.006849024444818497,
            0.0007423628703691065,
        };
        D_B = {
            0.0769348293542862,
            -0.0064767226576805115,
            0.00480955233797431,
            -0.0011584069579839706,
        };
        D_R = {
            0.05770963057875633,
            0.015429234132170677,
            -0.004901690874248743,
            5.2344439609441906e-05,
        };
    }

    distortion_params_front  = D_F;
    distortion_params_left   = D_L;
    distortion_params_behind = D_B;
    distortion_params_right  = D_R;
}

inline Eigen::MatrixXd getMatrix(double yaw, double pitch, double roll)
{
    double c_y = cos(yaw);
    double s_y = sin(yaw);
    double c_r = cos(roll);
    double s_r = sin(roll);
    double c_p = cos(pitch);
    double s_p = sin(pitch);
    Eigen::MatrixXd matrix(3, 3);
    matrix(0, 0) = c_p * c_y;
    matrix(0, 1) = c_y * s_p * s_r - s_y * c_r;
    matrix(0, 2) = c_y * s_p * c_r + s_y * s_r;
    matrix(1, 0) = s_y * c_p;
    matrix(1, 1) = s_y * s_p * s_r + c_y * c_r;
    matrix(1, 2) = s_y * s_p * c_r - c_y * s_r;
    matrix(2, 0) = -s_p;
    matrix(2, 1) = c_p * s_r;
    matrix(2, 2) = c_p * c_r;
    return matrix;
}

void Optimizer::initializePose()
{
    Eigen::Matrix4d T_FG;
    Eigen::Matrix4d T_LG;
    Eigen::Matrix4d T_BG;
    Eigen::Matrix4d T_RG;
    // mean: 1.88649
    // rvecs:
    // 0.00348309, 0.0371387, 2.36463
    // tvecs:
    // -781.234
    // 1436.17
    // 66.6156

    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        T_FG << 1, 0, 0, 0, 0, 0, 1, -4.1, 0, -1, 0, -2.5, 0, 0, 0, 1;
        T_LG << 0, -1, 0, 0, 0, 0, 1, -4.1, -1, 0, 0, -1, 0, 0, 0, 1;
        T_BG << -1, 0, 0, 0, 0, 0, 1, -4.1, 0, 1, 0, -2, 0, 0, 0, 1;
        T_RG << 0, 1, 0, 0, 0, 0, 1, -4.1, 1, 0, 0, -1, 0, 0, 0, 1;
    }

    if (data_index == "imgs1" || data_index == "imgs2")
    {
        // clang-format off
	auto r_mat = getMatrix(0.00348309, 0.0371387, 2.36463);
	T_FG << r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), -781.234/1000,
		r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), 1436.17/1000 ,
		r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), 66.6156/1000 ,
		0, 0, 0, 1;

	r_mat = getMatrix(1.63932, -1.89911, -0.0459019);
	T_RG << r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), -29.6463/1000,
		r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), 1070.4/1000  ,
		r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), -1909.5/1000 ,
		0, 0, 0, 1;

	r_mat = getMatrix(-1.54026, 2.19065, 0.0314631);
	T_LG << r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), 15.4238/1000 ,
		r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), 1670.21/1000 ,
		r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), -1244.62/1000,
		0, 0, 0, 1;

	r_mat = getMatrix(3.138, -0.0157633, -2.33495);
	T_BG << r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), 739.047/1000,
		r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), 1411.51/1000,
		r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), 11.8222/1000,
		0, 0, 0, 1;

        printf("Set custom Extrinsics\n");

        // T_FG << 9.99277118e-01, 3.82390286e-04, -3.80143958e-02, 6.75437418e-01 / 10, 
        //         -2.30748265e-02, -7.88582447e-01, -6.14495953e-01, 2.50896883e+01 / 10, 
        //         -3.02124625e-02, 6.14928921e-01, -7.88003572e-01, 3.17779305e+00 / 10, 
        //         0, 0, 0, 1;
        //
        // T_LG << -1.21898860e-02, 9.99924056e-01, -1.81349393e-03,
        //         1.36392943e+00 / 10, 8.02363600e-01, 8.69913885e-03,
        //         -5.96772133e-01, 1.60942881e+01 / 10, -5.96711036e-01,
        //         -8.72966581e-03, -8.02408707e-01, 1.04105913e+01 / 10, 
	       //  0, 0, 0, 1;
        //
        // T_BG << -9.99615699e-01, 1.56439861e-02, -2.28849354e-02,
        //         1.09266953e+00 / 10, 2.59906371e-02, 8.16008735e-01,
        //         -5.77454960e-01, 2.46308124e+01 / 10, 9.64060983e-03,
        //         -5.77827838e-01, -8.16101739e-01, 6.60957845e+00 / 10, 
	       //  0, 0, 0, 1;
        //
        // T_RG << 4.57647596e-03, -9.99989102e-01, 9.22798184e-04,
        //         -1.66115120e-01 / 10, -6.26343448e-01, -3.58584197e-03,
        //         -7.79538984e-01, 1.76226207e+01 / 10, 7.79533797e-01,
        //         2.98955282e-03, -6.26353033e-01, 6.08338205e+00 / 10, 
	       //  0, 0, 0, 1;
        // clang-format on
    }

    if (data_index == "imgs6")
    {
        // Ground<--Car_Center
        Eigen::Matrix4d T_GC;
        T_GC << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

        // Car_Center<--Front
        Eigen::Matrix4d T_CF;
        T_CF << 0.9997334480285645, -0.023079626262187958,
            0.0005962559371255338, -8.085761510301381e-05,
            -0.009539025835692883, -0.38940492272377014, 0.9210171699523926,
            3.7151906490325928 - 1.8, -0.021024547517299652,
            -0.9207773804664612, -0.3895213007926941, 0.6170634031295776, 0, 0,
            0, 1;
        T_FG = (T_GC * T_CF).inverse();

        // Car_Center<--Left
        Eigen::Matrix4d T_CL;
        T_CL << 0.005555491428822279, 0.5890335440635681, -0.8080894947052002,
            -0.9713481068611145, 0.9997195601463318, -0.021872058510780334,
            -0.009070112369954586, 2.1220741271972656 - 1.8,
            -0.023017220199108124, -0.8078124523162842, -0.5889899134635925,
            1.0284450054168701, 0, 0, 0, 1;
        T_LG = (T_GC * T_CL).inverse();

        // Car_Center<--Back
        Eigen::Matrix4d T_CB;
        T_CB << -0.9995912909507751, -0.022174330428242683,
            -0.01802690140902996, -0.003093475243076682, 0.005489660892635584,
            0.47006523609161377, -0.8826141953468323, -1.0302399396896362 - 1.8,
            0.028045203536748886, -0.882352888584137, -0.46975162625312805,
            0.70152747631073, 0, 0, 0, 1;
        T_BG = (T_GC * T_CB).inverse();

        // Car_Center<--Right
        Eigen::Matrix4d T_CR;
        T_CR << -0.01291694212704897, -0.5848034620285034, 0.811072051525116,
            0.9695128202438354, -0.9998561143875122, -0.0013508035335689783,
            -0.01689748652279377, 2.1217870712280273 - 1.8,
            0.010977385565638542, -0.8111737966537476, -0.5847020149230957,
            1.030495285987854, 0, 0, 0, 1;
        T_RG = (T_GC * T_CR).inverse();
    }

    if (flag_add_disturbance)
    {
        if (fixed == "back")
        {
            Eigen::Matrix4d front_disturbance;
            Eigen::Matrix3d front_disturbance_rot_mat;
            Vec3f front_disturbance_rot_euler;  // R(euler)
            Mat_<double> front_disturbance_t =
                (Mat_<double>(3, 1) << 0.007, 0.008, -0.0093);
            front_disturbance_rot_euler << 0.89, 2.69, 1.05;
            front_disturbance_rot_mat =
                TransformUtil::eulerAnglesToRotationMatrix(
                    front_disturbance_rot_euler);
            front_disturbance = TransformUtil::R_T2RT(
                TransformUtil::eigen2mat(front_disturbance_rot_mat),
                front_disturbance_t);
            T_FG *= front_disturbance;
        }

        Eigen::Matrix4d left_disturbance;
        Eigen::Matrix3d left_disturbance_rot_mat;
        Vec3f left_disturbance_rot_euler;  // R(euler)
        // Mat_<double> left_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
        Mat_<double> left_disturbance_t =
            (Mat_<double>(3, 1) << 0.0095, 0.0025, -0.0086);
        left_disturbance_rot_euler << 1.95, -1.25, 2.86;
        left_disturbance_rot_mat = TransformUtil::eulerAnglesToRotationMatrix(
            left_disturbance_rot_euler);
        left_disturbance = TransformUtil::R_T2RT(
            TransformUtil::eigen2mat(left_disturbance_rot_mat),
            left_disturbance_t);
        T_LG *= left_disturbance;

        Eigen::Matrix4d right_disturbance;
        Eigen::Matrix3d right_disturbance_rot_mat;
        Vec3f right_disturbance_rot_euler;
        // Mat_<double> right_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
        Mat_<double> right_disturbance_t =
            (Mat_<double>(3, 1) << 0.0065, -0.0075, 0.0095);
        right_disturbance_rot_euler << 2.95, 0.95, -1.8;
        right_disturbance_rot_mat = TransformUtil::eulerAnglesToRotationMatrix(
            right_disturbance_rot_euler);
        right_disturbance = TransformUtil::R_T2RT(
            TransformUtil::eigen2mat(right_disturbance_rot_mat),
            right_disturbance_t);
        T_RG *= right_disturbance;

        if (fixed == "front")
        {
            Eigen::Matrix4d behind_disturbance;
            Eigen::Matrix3d behind_disturbance_rot_mat;
            Vec3f behind_disturbance_rot_euler;
            // Mat_<double> behind_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
            Mat_<double> behind_disturbance_t =
                (Mat_<double>(3, 1) << -0.002, -0.0076, 0.0096);
            behind_disturbance_rot_euler << -1.75, 2.95, -1.8;
            behind_disturbance_rot_mat =
                TransformUtil::eulerAnglesToRotationMatrix(
                    behind_disturbance_rot_euler);
            behind_disturbance = TransformUtil::R_T2RT(
                TransformUtil::eigen2mat(behind_disturbance_rot_mat),
                behind_disturbance_t);
            T_BG *= behind_disturbance;
        }
    }

    extrinsic_front  = T_FG;
    extrinsic_left   = T_LG;
    extrinsic_behind = T_BG;
    extrinsic_right  = T_RG;

    cout << "extrinsic_front:" << endl << extrinsic_front << endl;
    cout << "euler:" << endl
         << TransformUtil::Rotation2Eul(extrinsic_front.block(0, 0, 3, 3))
         << endl;
    cout << "extrinsic_left:" << endl << extrinsic_left << endl;
    cout << "euler:" << endl
         << TransformUtil::Rotation2Eul(extrinsic_left.block(0, 0, 3, 3))
         << endl;
    cout << "extrinsic_right:" << endl << extrinsic_right << endl;
    cout << "euler:" << endl
         << TransformUtil::Rotation2Eul(extrinsic_right.block(0, 0, 3, 3))
         << endl;
    cout << "extrinsic_behind:" << endl << extrinsic_behind << endl;
    cout << "euler:" << endl
         << TransformUtil::Rotation2Eul(extrinsic_behind.block(0, 0, 3, 3))
         << endl;
    return;
}

void Optimizer::initializeKG()
{
    Eigen::Matrix3d K_G = Eigen::Matrix3d::Zero();

    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        K_G << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750,
            0.00000000, 0.00000000, 1.00000000;
        KG = K_G;
    }

    if (data_index == "imgs1" || data_index == "imgs2")
    {
        K_G(0, 0) = 421;
        K_G(1, 1) = -421;
        K_G(0, 2) = bcols / 2;
        K_G(1, 2) = brows / 2;
        K_G(2, 2) = 1.0;
        KG        = K_G;
    }

    if (data_index == "imgs6")
    {
        K_G(0, 0) = 500;
        K_G(1, 1) = 500;
        K_G(0, 2) = bcols / 2;
        K_G(1, 2) = brows / 2;
        K_G(2, 2) = 1.0;
        KG        = K_G;
    }
}

void Optimizer::initializeHeight()
{
    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        hf = 5.1;
        hl = 5.1;
        hb = 5.1;
        hr = 5.1;
    }
    else if (data_index == "imgs1" || data_index == "imgs2")
    {
        hf = 6.0;
        hl = 6.0;
        hb = 6.0;
        hr = 6.0;
    }
    else if (data_index == "imgs6")
    {
        hf = 4.0;
        hl = 4.0;
        hb = 4.0;
        hr = 4.0;
    }
}

void Optimizer::initializetailsize()
{
    if (data_index == "imgs3" || data_index == "imgs4" || data_index == "imgs5")
    {
        sizef = 450;
        sizel = 450;
        sizeb = 350;
        sizer = 450;
    }

    if (data_index == "imgs1" || data_index == "imgs2")
    {
        sizef = 420;
        sizel = 380;
        sizeb = 420;
        sizer = 360;
    }

    if (data_index == "imgs6")
    {
        sizef = 250;
        sizel = 350;
        sizeb = 150;
        sizer = 380;
    }
}

Optimizer::Optimizer(const Mat *imgf, const Mat *imgl, const Mat *imgb,
                     const Mat *imgr, int camera_model_index, int rows,
                     int cols, string fixed_, int flag, string data_set,
                     int _flag_add_disturbance, string prefix_,
                     string solution_model_ = "atb+gray")
{
    prefix    = prefix_;
    imgf_rgb  = *imgf;
    imgf_gray = gray_gamma(imgf_rgb);
    imgf_atb  = gray_atb(imgf_rgb);
    // imwrite(this->prefix+"/front_atb.png",imgf_atb);

    imgl_rgb  = *imgl;
    imgl_gray = gray_gamma(imgl_rgb);
    imgl_atb  = gray_atb(imgl_rgb);
    // imwrite(this->prefix+"/left_atb.png",imgl_atb);

    imgb_rgb  = *imgb;
    imgb_gray = gray_gamma(imgb_rgb);
    imgb_atb  = gray_atb(imgb_rgb);
    // imwrite(this->prefix+"/back_atb.png",imgb_atb);

    imgr_rgb  = *imgr;
    imgr_gray = gray_gamma(imgr_rgb);
    imgr_atb  = gray_atb(imgr_rgb);
    // imwrite(this->prefix+"/right_atb.png",imgr_atb);

    brows = rows;
    bcols = cols;

    data_index           = data_set;
    flag_add_disturbance = _flag_add_disturbance;

    initializeK();
    initializeD();
    initializeHeight();
    initializePose();
    initializeKG();
    initializetailsize();

    solution_model = solution_model_;

    camera_model = camera_model_index;

    bestVal_.resize(3, vector<double>(6));

    fixed = fixed_;

    coarse_flag = flag;

    if (fixed == "front")
    {
        imgf_bev_gray =
            project_on_ground(imgf_gray, extrinsic_front, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_atb =
            project_on_ground(imgf_gray, extrinsic_front, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_rgb =
            project_on_ground(imgf_rgb, extrinsic_front, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_gray = tail(imgf_bev_gray, "f");
        imgf_bev_atb  = tail(imgf_bev_atb, "f");
        imgf_bev_rgb  = tail(imgf_bev_rgb, "f");
    }
    else
    {
        imgb_bev_gray =
            project_on_ground(imgb_gray, extrinsic_behind, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_atb =
            project_on_ground(imgb_rgb, extrinsic_behind, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_rgb =
            project_on_ground(imgb_rgb, extrinsic_behind, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_gray = tail(imgb_bev_gray, "b");
        imgb_bev_atb  = tail(imgb_bev_atb, "b");
        imgb_bev_rgb  = tail(imgb_bev_rgb, "b");
    }
}

Optimizer::~Optimizer() {}

Mat Optimizer::project_on_ground(Mat img, Eigen::Matrix4d T_CG,
                                 Eigen::Matrix3d K_C, vector<double> D_C,
                                 Eigen::Matrix3d K_G, int rows, int cols,
                                 float height)
{
    Mat p_G = Mat::ones(3, rows * cols, CV_64FC1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            p_G.at<double>(0, cols * i + j) = j;
            p_G.at<double>(1, cols * i + j) = i;
        }
    }

    Mat P_G                         = Mat::ones(4, rows * cols, CV_64FC1);
    P_G(Rect(0, 0, rows * cols, 3)) = eigen2mat(K_G.inverse()) * p_G * height;
    if (camera_model == 0) P_G(Rect(0, 2, rows * cols, 1)) = 0;
    Mat P_GC  = Mat::zeros(4, rows * cols, CV_64FC1);
    Mat T_CG_ = (Mat_<double>(4, 4) << T_CG(0, 0), T_CG(0, 1), T_CG(0, 2),
                 T_CG(0, 3), T_CG(1, 0), T_CG(1, 1), T_CG(1, 2), T_CG(1, 3),
                 T_CG(2, 0), T_CG(2, 1), T_CG(2, 2), T_CG(2, 3), T_CG(3, 0),
                 T_CG(3, 1), T_CG(3, 2), T_CG(3, 3));
    P_GC      = T_CG_ * P_G;

    Mat P_GC1 = Mat::zeros(1, rows * cols, CV_64FC2);
    vector<Mat> channels(2);
    split(P_GC1, channels);
    channels[0] =
        P_GC(Rect(0, 0, rows * cols, 1)) / P_GC(Rect(0, 2, rows * cols, 1));
    channels[1] =
        P_GC(Rect(0, 1, rows * cols, 1)) / P_GC(Rect(0, 2, rows * cols, 1));
    merge(channels, P_GC1);

    Mat p_GC = Mat::zeros(1, rows * cols, CV_64FC2);
    Mat K_C_ =
        (Mat_<double>(3, 3) << K_C(0, 0), K_C(0, 1), K_C(0, 2), K_C(1, 0),
         K_C(1, 1), K_C(1, 2), K_C(2, 0), K_C(2, 1), K_C(2, 2));

    if (camera_model == 0)
    {
        fisheye::distortPoints(P_GC1, p_GC, K_C_, D_C);  // fisheye
    }
    else if (camera_model == 1)
    {  // Ocam
        distortPointsOcam(P_GC1, p_GC, K_C, D_C);
    }
    else
    {
        distortPoints(P_GC1, p_GC, K_C);  // pinhole
    }

    p_GC.reshape(rows, cols);
    Mat p_GC_table = p_GC.reshape(0, rows);
    Mat p_GC_table_32F;
    p_GC_table.convertTo(p_GC_table_32F, CV_32FC2);

    Mat img_GC;
    remap(img, img_GC, p_GC_table_32F, Mat(), INTER_LINEAR);
    return img_GC;
}

// perspective stiching
Mat Optimizer::generate_surround_view(Mat img_GF, Mat img_GL, Mat img_GB,
                                      Mat img_GR)
{
    Mat dst1, dst2, dst3;
    addWeighted(img_GF, 0.5, img_GL, 0.5, 3, dst1);
    addWeighted(dst1, 1.0, img_GB, 0.5, 3, dst2);
    addWeighted(dst2, 1.0, img_GR, 0.5, 3, dst3);
    return dst3;
}

// triangle stiching
Mat Optimizer::generate_surround_viewX(cv::Mat img_GF, cv::Mat img_GL,
                                       cv::Mat img_GB, cv::Mat img_GR)
{
    cv::Mat img_G(brows, bcols, CV_8UC3);
    for (int i = 0; i < brows; i++)
    {
        for (int j = 0; j < bcols; j++)
        {
            if (i > j)
            {
                if (i > -j + brows)
                {
                    img_G.at<cv::Vec3b>(i, j) = img_GB.at<cv::Vec3b>(i, j);
                }
                else
                {
                    img_G.at<cv::Vec3b>(i, j) = img_GL.at<cv::Vec3b>(i, j);
                }
            }
            else
            {
                if (i > -j + bcols)
                {
                    img_G.at<cv::Vec3b>(i, j) = img_GR.at<cv::Vec3b>(i, j);
                }
                else
                {
                    img_G.at<cv::Vec3b>(i, j) = img_GF.at<cv::Vec3b>(i, j);
                }
            }
        }
    }
    return img_G;
}

double Optimizer::CostFunction(const vector<double> var, string idx,
                               Eigen::Matrix4d T)
{
    double loss;
    if (idx == "right")
    {
        Eigen::Matrix4d Tr     = T;
        Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
        Tr *= deltaT;
        if (fixed == "front")
            loss = back_camera_and_compute_loss(imgf_bev_gray, imgf_bev_atb,
                                                imgr_gray, imgr_atb, Tr, "fr");
        else  // fixed back,so rear bev pixels back projected to right camera
            loss = back_camera_and_compute_loss(imgb_bev_gray, imgb_bev_atb,
                                                imgr_gray, imgr_atb, Tr, "br");
        return loss;
    }
    else if (idx == "left")
    {
        Eigen::Matrix4d Tl     = T;
        Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
        Tl *= deltaT;
        if (fixed == "front")
            loss = back_camera_and_compute_loss(imgf_bev_gray, imgf_bev_atb,
                                                imgl_gray, imgl_atb, Tl, "fl");
        else  // fixed back,so rear bev pixels back projected to left camera
            loss = back_camera_and_compute_loss(imgb_bev_gray, imgb_bev_atb,
                                                imgl_gray, imgl_atb, Tl, "bl");
        return loss;
    }
    else
    {  // behind(fist_order="front") or front(fist_order="behind")
        Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
        if (fixed == "front")
        {
            Eigen::Matrix4d Tb = T;
            Tb *= deltaT;
            loss = back_camera_and_compute_loss(imgl_bev_gray, imgl_bev_atb,
                                                imgb_gray, imgb_atb, Tb, "lb");
            loss += back_camera_and_compute_loss(imgr_bev_gray, imgr_bev_atb,
                                                 imgb_gray, imgb_atb, Tb, "rb");
        }
        else
        {  // fixed back,so left&right bev pixels back projected to front
           // camera at last
            Eigen::Matrix4d Tf = T;
            Tf *= deltaT;
            loss = back_camera_and_compute_loss(imgl_bev_gray, imgl_bev_atb,
                                                imgf_gray, imgf_atb, Tf, "lf");
            loss += back_camera_and_compute_loss(imgr_bev_gray, imgr_bev_atb,
                                                 imgf_gray, imgf_atb, Tf, "rf");
        }
        return loss;
    }
}

double Optimizer::back_camera_and_compute_loss(Mat img1_bev_gray,
                                               Mat img1_bev_atb, Mat img2_gray,
                                               Mat img2_atb, Eigen::Matrix4d T,
                                               string idx)
{
    vector<pair<cv::Point, double>> pixels;
    Eigen::Matrix3d KC;
    vector<double> DC;
    Mat pG;
    Mat PG;
    Mat show;
    double ncoef;
    if (idx == "fl")
    {  // bev_front->camera_left
        // show=imgl_rgb.clone();  //when test and display back projection
        // process, Mat(show) is needed
        DC = distortion_params_left;  // distortortion params whose camera is
                                      // back projected to
        KC     = intrinsic_left;      // K whose camera is back projected to
        pG     = pG_fl;               // front bev texture pixels(Mat)
        PG     = PG_fl;               // front bev pixels->3d points(Mat)
        pixels = fl_pixels_texture;   // front bev texture pixels(vector)
        ncoef  = ncoef_fl;  // mean gray luminosity of front bev in commonview /
                            // mean gray luminosity of left bev in commonview
    }
    else if (idx == "fr")
    {
        // show=imgr_rgb.clone();
        // show=imgr_gray.clone();
        // cvtColor(show,show,CV_GRAY2BGR);
        DC     = distortion_params_right;
        KC     = intrinsic_right;
        pG     = pG_fr;
        PG     = PG_fr;
        pixels = fr_pixels_texture;
        ncoef  = ncoef_fr;
    }
    else if (idx == "lb")
    {
        // show=imgb_rgb.clone();
        DC     = distortion_params_behind;
        KC     = intrinsic_behind;
        pG     = pG_bl;
        PG     = PG_bl;
        pixels = bl_pixels_texture;
        ncoef  = ncoef_bl;
    }
    else if (idx == "rb")
    {
        // show=imgb_rgb.clone();
        DC     = distortion_params_behind;
        KC     = intrinsic_behind;
        pG     = pG_br;
        PG     = PG_br;
        pixels = br_pixels_texture;
        ncoef  = ncoef_br;
    }
    else if (idx == "lf")
    {
        // show=imgf_rgb.clone();
        DC     = distortion_params_front;
        KC     = intrinsic_front;
        pG     = pG_fl;
        PG     = PG_fl;
        pixels = fl_pixels_texture;
        ncoef  = ncoef_fl;
    }
    else if (idx == "rf")
    {
        // show=imgf_rgb.clone();
        DC     = distortion_params_front;
        KC     = intrinsic_front;
        pG     = pG_fr;
        PG     = PG_fr;
        pixels = fr_pixels_texture;
        ncoef  = ncoef_fr;
    }
    else if (idx == "br")
    {
        // show=imgr_rgb.clone();
        DC     = distortion_params_right;
        KC     = intrinsic_right;
        pG     = pG_br;
        PG     = PG_br;
        pixels = br_pixels_texture;
        ncoef  = ncoef_br;
    }
    else if (idx == "bl")
    {
        // show=imgl_rgb.clone();
        DC     = distortion_params_left;
        KC     = intrinsic_left;
        pG     = pG_bl;
        PG     = PG_bl;
        pixels = bl_pixels_texture;
        ncoef  = ncoef_bl;
    }

    int size = pixels.size();
    if (camera_model == 0)
        PG(Rect(0, 2, size, 1)) =
            0;  // !!! 不同版本opencv可能有差异---cv::Mat::zeros(size, 1,
                // CV_64FC1);
    double loss   = 0;
    int failcount = 0;
    Mat PG2C      = Mat::zeros(4, size, CV_64FC1);
    PG2C          = eigen2mat(T) * PG;
    Mat PG2C1     = Mat::zeros(1, size, CV_64FC2);
    vector<Mat> channels(2);
    split(PG2C1, channels);
    channels[0] = PG2C(Rect(0, 0, size, 1)) / PG2C(Rect(0, 2, size, 1));
    channels[1] = PG2C(Rect(0, 1, size, 1)) / PG2C(Rect(0, 2, size, 1));
    merge(channels, PG2C1);
    Mat pG2C(1, size, CV_64FC2);
    if (camera_model == 0)
        fisheye::distortPoints(PG2C1, pG2C, eigen2mat(KC), DC);
    else if (camera_model == 1)
        distortPointsOcam(PG2C1, pG2C, KC, DC);
    else
        distortPoints(PG2C1, pG2C, KC);
    for (int i = 0; i < size; i++)
    {
        double x  = pG.at<double>(0, i);
        double y  = pG.at<double>(1, i);
        double x1 = pG2C.at<Vec2d>(0, i)[0];
        double y1 = pG2C.at<Vec2d>(0, i)[1];
        // cout<<x1<<" "<<y1<<endl;
        if (x1 > 0 && y1 > 0 && x1 < img2_gray.cols && y1 < img2_gray.rows)
        {
            if (solution_model == "atb+gray")
            {
                if (phase == 1)
                    loss += sqrt(pow(fabs(getPixelValue(&img1_bev_atb, x, y) -
                                          getPixelValue(&img2_atb, x1, y1)),
                                     2));
                else
                    loss += sqrt(
                        pow(fabs(getPixelValue(&img1_bev_gray, x, y) / ncoef -
                                 getPixelValue(&img2_gray, x1, y1)) *
                                pixels[i].second,
                            2));
                // if(idx=="fl")
                // 	circle(show,Point(x1,y1),1,Scalar(255,0,0),-1);
            }
            else if (solution_model == "gray")
            {
                loss +=
                    sqrt(pow(fabs(getPixelValue(&img1_bev_gray, x, y) / ncoef -
                                  getPixelValue(&img2_gray, x1, y1)) *
                                 pixels[i].second,
                             2));
            }
            else
            {  // solution_model=="atb"
                loss += sqrt(pow(fabs(getPixelValue(&img1_bev_atb, x, y) -
                                      getPixelValue(&img2_atb, x1, y1)),
                                 2));
            }
        }
        else
        {
            failcount++;
            if (failcount > 30) return INT_MAX;
        }
    }

    // ------test and display back projection process------
    // if(idx=="fl"){
    // 	imshow("back pixels",show);
    // 	waitKey(0);
    // }

    return loss;
}

void Optimizer::random_search_params(int search_count, double roll_ep0,
                                     double roll_ep1, double pitch_ep0,
                                     double pitch_ep1, double yaw_ep0,
                                     double yaw_ep1, double t0_ep0,
                                     double t0_ep1, double t1_ep0,
                                     double t1_ep1, double t2_ep0,
                                     double t2_ep1, string idx)
{
    vector<double> var(6, 0.0);
    double resolution_r = 100;  // resolution_r could be customized to get
                                // better result in different scenes
    double resolution_t = 100;  // resolution_t could be customized to get
                                // better result in different scenes

    random_device generator;
    std::uniform_int_distribution<int> distribution_roll(
        roll_ep0 * resolution_r, roll_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_pitch(
        pitch_ep0 * resolution_r, pitch_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_yaw(yaw_ep0 * resolution_r,
                                                        yaw_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_x(t0_ep0 * resolution_t,
                                                      t0_ep1 * resolution_t);
    std::uniform_int_distribution<int> distribution_y(t1_ep0 * resolution_t,
                                                      t1_ep1 * resolution_t);
    std::uniform_int_distribution<int> distribution_z(t2_ep0 * resolution_t,
                                                      t2_ep1 * resolution_t);

    for (size_t i = 0; i < search_count; i++)
    {
        mutexval.lock();
        var[0] = double(distribution_roll(generator)) / resolution_r;
        var[1] = double(distribution_pitch(generator)) / resolution_r;
        var[2] = double(distribution_yaw(generator)) / resolution_r;
        var[3] = double(distribution_x(generator)) / resolution_t;
        var[4] = double(distribution_y(generator)) / resolution_t;
        var[5] = double(distribution_z(generator)) / resolution_t;
        mutexval.unlock();

        if (idx == "left")
        {
            double loss_new = CostFunction(var, idx, extrinsic_left);
            if (loss_new < cur_left_loss)
            {
                lock_guard<std::mutex> lock(mutexleft);
                cur_left_loss = loss_new;
                extrinsic_left_opt =
                    extrinsic_left * TransformUtil::GetDeltaT(var);
                bestVal_[0] = var;
            }
        }
        if (idx == "right")
        {
            double loss_new = CostFunction(var, idx, extrinsic_right);
            if (loss_new < cur_right_loss)
            {
                lock_guard<std::mutex> lock(mutexright);
                cur_right_loss = loss_new;
                extrinsic_right_opt =
                    extrinsic_right * TransformUtil::GetDeltaT(var);
                bestVal_[1] = var;
            }
        }
        if (idx == "behind")
        {
            double loss_new = CostFunction(var, idx, extrinsic_behind);
            if (loss_new < cur_behind_loss)
            {
                lock_guard<std::mutex> lock(mutexbehind);
                cur_behind_loss = loss_new;
                extrinsic_behind_opt =
                    extrinsic_behind * TransformUtil::GetDeltaT(var);
                bestVal_[2] = var;
            }
        }
        if (idx == "front")
        {  // if fix back camera,front camera is calibrated at last
            double loss_new = CostFunction(var, idx, extrinsic_front);
            if (loss_new < cur_front_loss)
            {
                lock_guard<std::mutex> lock(mutexfront);
                cur_front_loss = loss_new;
                extrinsic_front_opt =
                    extrinsic_front * TransformUtil::GetDeltaT(var);
                bestVal_[2] = var;
            }
        }
    }

    if (idx == "left")
    {
        lock_guard<std::mutex> lock(mutexleft);
        imgl_bev_gray =
            project_on_ground(imgl_gray, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_rgb =
            project_on_ground(imgl_rgb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_atb =
            project_on_ground(imgl_atb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);

        imgl_bev_gray = tail(imgl_bev_gray, "l");
        imgl_bev_rgb  = tail(imgl_bev_rgb, "l");
        imgl_bev_atb  = tail(imgl_bev_atb, "l");

        if (fixed == "front")
        {
            Mat mask = Binarization(imgf_bev_rgb, imgl_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgf_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgl_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_fl     = mean1 / mean2;
        }
        else
        {
            Mat mask = Binarization(imgb_bev_rgb, imgl_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgb_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgl_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_bl     = mean1 / mean2;
        }
    }
    else if (idx == "right")
    {
        lock_guard<std::mutex> lock(mutexright);
        imgr_bev_gray =
            project_on_ground(imgr_gray, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_rgb =
            project_on_ground(imgr_rgb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_atb =
            project_on_ground(imgr_atb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);

        imgr_bev_gray = tail(imgr_bev_gray, "r");
        imgr_bev_rgb  = tail(imgr_bev_rgb, "r");
        imgr_bev_atb  = tail(imgr_bev_atb, "r");

        if (fixed == "front")
        {
            Mat mask = Binarization(imgf_bev_rgb, imgr_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgf_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgr_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_fr     = mean1 / mean2;
        }
        else
        {
            Mat mask = Binarization(imgb_bev_rgb, imgr_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgb_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgr_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_br     = mean1 / mean2;
        }
    }
    else if (idx == "behind")
    {
        lock_guard<std::mutex> lock(mutexbehind);
        imgb_bev_gray =
            project_on_ground(imgb_gray, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_rgb =
            project_on_ground(imgb_rgb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_atb =
            project_on_ground(imgb_atb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);

        imgb_bev_gray = tail(imgb_bev_gray, "b");
        imgb_bev_rgb  = tail(imgb_bev_rgb, "b");
        imgb_bev_atb  = tail(imgb_bev_atb, "b");

        Mat mask1 = Binarization(imgl_bev_rgb, imgb_bev_rgb);
        Mat ROI1, ROI2;
        bitwise_and(imgl_bev_gray, mask1, ROI1);
        // imshow("ROI1",ROI1);
        // waitKey(0);
        bitwise_and(imgb_bev_gray, mask1, ROI2);
        // imshow("ROI2",ROI2);
        // waitKey(0);
        double mean1 = mean(ROI1).val[0];
        double mean2 = mean(ROI2).val[0];
        ncoef_bl     = mean1 / mean2;

        Mat mask2 = Binarization(imgr_bev_rgb, imgb_bev_rgb);
        Mat ROI3, ROI4;
        bitwise_and(imgr_bev_gray, mask2, ROI3);
        // imshow("ROI3",ROI3);
        // waitKey(0);
        bitwise_and(imgb_bev_gray, mask2, ROI4);
        // imshow("ROI4",ROI4);
        // waitKey(0);
        double mean3 = mean(ROI3).val[0];
        double mean4 = mean(ROI4).val[0];
        ncoef_br     = mean3 / mean4;
    }
    else
    {  // if fix back camera,front camera is calibrated at last
        lock_guard<std::mutex> lock(mutexfront);
        imgf_bev_gray =
            project_on_ground(imgf_gray, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_rgb =
            project_on_ground(imgf_rgb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);
        imgf_bev_atb =
            project_on_ground(imgf_atb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);

        imgf_bev_gray = tail(imgf_bev_gray, "f");
        imgf_bev_rgb  = tail(imgf_bev_rgb, "f");
        imgf_bev_atb  = tail(imgf_bev_atb, "f");

        Mat mask1 = Binarization(imgl_bev_rgb, imgf_bev_rgb);
        Mat ROI1, ROI2;
        bitwise_and(imgl_bev_gray, mask1, ROI1);
        // imshow("ROI1",ROI1);
        // waitKey(0);
        bitwise_and(imgf_bev_gray, mask1, ROI2);
        // imshow("ROI2",ROI2);
        // waitKey(0);
        double mean1 = mean(ROI1).val[0];
        double mean2 = mean(ROI2).val[0];
        ncoef_fl     = mean1 / mean2;

        Mat mask2 = Binarization(imgr_bev_rgb, imgf_bev_rgb);
        Mat ROI3, ROI4;
        bitwise_and(imgr_bev_gray, mask2, ROI3);
        // imshow("ROI3",ROI3);
        // waitKey(0);
        bitwise_and(imgf_bev_gray, mask2, ROI4);
        // imshow("ROI4",ROI4);
        // waitKey(0);
        double mean3 = mean(ROI3).val[0];
        double mean4 = mean(ROI4).val[0];
        ncoef_fr     = mean3 / mean4;
    }
}

void Optimizer::fine_random_search_params(int search_count, double roll_ep0,
                                          double roll_ep1, double pitch_ep0,
                                          double pitch_ep1, double yaw_ep0,
                                          double yaw_ep1, double t0_ep0,
                                          double t0_ep1, double t1_ep0,
                                          double t1_ep1, double t2_ep0,
                                          double t2_ep1, string idx)
{
    vector<double> var(6, 0.0);
    double resolution_r = 200;  // resolution_r could be customized to get
                                // better result in different scenes
    double resolution_t = 200;  // resolution_t could be customized to get
                                // better result in different scenes

    random_device generator;
    std::uniform_int_distribution<int> distribution_roll(
        roll_ep0 * resolution_r, roll_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_pitch(
        pitch_ep0 * resolution_r, pitch_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_yaw(yaw_ep0 * resolution_r,
                                                        yaw_ep1 * resolution_r);
    std::uniform_int_distribution<int> distribution_x(t0_ep0 * resolution_t,
                                                      t0_ep1 * resolution_t);
    std::uniform_int_distribution<int> distribution_y(t1_ep0 * resolution_t,
                                                      t1_ep1 * resolution_t);
    std::uniform_int_distribution<int> distribution_z(t2_ep0 * resolution_t,
                                                      t2_ep1 * resolution_t);

    // or search in real-number-field
    // std::uniform_real_distribution<double>
    // distribution_roll(roll_ep0,roll_ep1);
    // std::uniform_real_distribution<double>
    // distribution_pitch(pitch_ep0,pitch_ep1);
    // std::uniform_real_distribution<double> distribution_yaw(yaw_ep0,yaw_ep1);
    // std::uniform_real_distribution<double> distribution_x(t0_ep0, t0_ep1);
    // std::uniform_real_distribution<double> distribution_y(t1_ep0, t1_ep1);
    // std::uniform_real_distribution<double> distribution_z(t2_ep0, t2_ep1);

    for (size_t i = 0; i < search_count; i++)
    {
        mutexval.lock();
        var[0] = double(distribution_roll(generator)) / resolution_r;
        var[1] = double(distribution_pitch(generator)) / resolution_r;
        var[2] = double(distribution_yaw(generator)) / resolution_r;
        var[3] = double(distribution_x(generator)) / resolution_t;
        var[4] = double(distribution_y(generator)) / resolution_t;
        var[5] = double(distribution_z(generator)) / resolution_t;

        // search in real-number-field
        // var[0] = distribution_roll(generator);
        // var[1] = distribution_pitch(generator);
        // var[2] = distribution_yaw(generator);
        // var[3] = distribution_x(generator);
        // var[4] = distribution_y(generator);
        // var[5] = distribution_z(generator);
        mutexval.unlock();

        if (idx == "left")
        {
            double loss_new = CostFunction(var, idx, extrinsic_left_opt);
            if (loss_new < cur_left_loss)
            {
                lock_guard<std::mutex> lock(mutexleft);
                cur_left_loss = loss_new;
                extrinsic_left_opt =
                    extrinsic_left_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[0][i] += var[i];
                }
            }
        }
        if (idx == "right")
        {
            double loss_new = CostFunction(var, idx, extrinsic_right_opt);
            if (loss_new < cur_right_loss)
            {
                lock_guard<std::mutex> lock(mutexright);
                cur_right_loss = loss_new;
                extrinsic_right_opt =
                    extrinsic_right_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[1][i] += var[i];
                }
            }
        }
        if (idx == "behind")
        {
            double loss_new = CostFunction(var, idx, extrinsic_behind_opt);
            if (loss_new < cur_behind_loss)
            {
                lock_guard<std::mutex> lock(mutexbehind);
                cur_behind_loss = loss_new;
                extrinsic_behind_opt =
                    extrinsic_behind_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[2][i] += var[i];
                }
            }
        }
        if (idx == "front")
        {  // if fix back camera,front camera is calibrated at last
            double loss_new = CostFunction(var, idx, extrinsic_front_opt);
            if (loss_new < cur_front_loss)
            {
                lock_guard<std::mutex> lock(mutexfront);
                cur_front_loss = loss_new;
                extrinsic_front_opt =
                    extrinsic_front_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[2][i] += var[i];
                }
            }
        }
    }

    if (idx == "left")
    {
        lock_guard<std::mutex> lock(mutexleft);
        imgl_bev_gray =
            project_on_ground(imgl_gray, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_rgb =
            project_on_ground(imgl_rgb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_atb =
            project_on_ground(imgl_atb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);

        imgl_bev_gray = tail(imgl_bev_gray, "l");
        imgl_bev_rgb  = tail(imgl_bev_rgb, "l");
        imgl_bev_atb  = tail(imgl_bev_atb, "l");

        if (fixed == "front")
        {
            Mat mask = Binarization(imgf_bev_rgb, imgl_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgf_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgl_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_fl     = mean1 / mean2;
        }
        else
        {
            Mat mask = Binarization(imgb_bev_rgb, imgl_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgb_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgl_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_bl     = mean1 / mean2;
        }
    }
    else if (idx == "right")
    {
        lock_guard<std::mutex> lock(mutexright);
        imgr_bev_gray =
            project_on_ground(imgr_gray, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_rgb =
            project_on_ground(imgr_rgb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_atb =
            project_on_ground(imgr_atb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);

        imgr_bev_gray = tail(imgr_bev_gray, "r");
        imgr_bev_rgb  = tail(imgr_bev_rgb, "r");
        imgr_bev_atb  = tail(imgr_bev_atb, "r");

        if (fixed == "front")
        {
            Mat mask = Binarization(imgf_bev_rgb, imgr_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgf_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgr_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_fr     = mean1 / mean2;
        }
        else
        {
            Mat mask = Binarization(imgb_bev_rgb, imgr_bev_rgb);
            Mat ROI1, ROI2;
            bitwise_and(imgb_bev_gray, mask, ROI1);
            // imshow("ROI1",ROI1);
            // waitKey(0);
            bitwise_and(imgr_bev_gray, mask, ROI2);
            // imshow("ROI2",ROI2);
            // waitKey(0);
            double mean1 = mean(ROI1).val[0];
            double mean2 = mean(ROI2).val[0];
            ncoef_br     = mean1 / mean2;
        }
    }
    else if (idx == "behind")
    {
        lock_guard<std::mutex> lock(mutexbehind);
        imgb_bev_gray =
            project_on_ground(imgb_gray, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_rgb =
            project_on_ground(imgb_rgb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_atb =
            project_on_ground(imgb_atb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);

        imgb_bev_gray = tail(imgb_bev_gray, "b");
        imgb_bev_rgb  = tail(imgb_bev_rgb, "b");
        imgb_bev_atb  = tail(imgb_bev_atb, "b");

        Mat mask1 = Binarization(imgl_bev_rgb, imgb_bev_rgb);
        Mat ROI1, ROI2;
        bitwise_and(imgl_bev_gray, mask1, ROI1);
        // imshow("ROI1",ROI1);
        // waitKey(0);
        bitwise_and(imgb_bev_gray, mask1, ROI2);
        // imshow("ROI2",ROI2);
        // waitKey(0);
        double mean1 = mean(ROI1).val[0];
        double mean2 = mean(ROI2).val[0];
        ncoef_bl     = mean1 / mean2;

        Mat mask2 = Binarization(imgr_bev_rgb, imgb_bev_rgb);
        Mat ROI3, ROI4;
        bitwise_and(imgr_bev_gray, mask2, ROI3);
        // imshow("ROI3",ROI3);
        // waitKey(0);
        bitwise_and(imgb_bev_gray, mask2, ROI4);
        // imshow("ROI4",ROI4);
        // waitKey(0);
        double mean3 = mean(ROI3).val[0];
        double mean4 = mean(ROI4).val[0];
        ncoef_br     = mean3 / mean4;
    }
    else
    {  // if fix back camera,front camera is calibrated at last
        lock_guard<std::mutex> lock(mutexfront);
        imgf_bev_gray =
            project_on_ground(imgf_gray, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_rgb =
            project_on_ground(imgf_rgb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);
        imgf_bev_atb =
            project_on_ground(imgf_atb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);

        imgf_bev_gray = tail(imgf_bev_gray, "f");
        imgf_bev_rgb  = tail(imgf_bev_rgb, "f");
        imgf_bev_atb  = tail(imgf_bev_atb, "f");

        Mat mask1 = Binarization(imgl_bev_rgb, imgf_bev_rgb);
        Mat ROI1, ROI2;
        bitwise_and(imgl_bev_gray, mask1, ROI1);
        // imshow("ROI1",ROI1);
        // waitKey(0);
        bitwise_and(imgf_bev_gray, mask1, ROI2);
        // imshow("ROI2",ROI2);
        // waitKey(0);
        double mean1 = mean(ROI1).val[0];
        double mean2 = mean(ROI2).val[0];
        ncoef_fl     = mean1 / mean2;

        Mat mask2 = Binarization(imgr_bev_rgb, imgf_bev_rgb);
        Mat ROI3, ROI4;
        bitwise_and(imgr_bev_gray, mask2, ROI3);
        // imshow("ROI3",ROI3);
        // waitKey(0);
        bitwise_and(imgf_bev_gray, mask2, ROI4);
        // imshow("ROI4",ROI4);
        // waitKey(0);
        double mean3 = mean(ROI3).val[0];
        double mean4 = mean(ROI4).val[0];
        ncoef_fr     = mean3 / mean4;
    }
}

void Optimizer::best_random_search_params(int search_count, double roll_ep0,
                                          double roll_ep1, double pitch_ep0,
                                          double pitch_ep1, double yaw_ep0,
                                          double yaw_ep1, double t0_ep0,
                                          double t0_ep1, double t1_ep0,
                                          double t1_ep1, double t2_ep0,
                                          double t2_ep1, string idx)
{
    vector<double> var(6, 0.0);
    // double resolution_r=200;
    // double resolution_t=200;

    // search in real-number-field
    random_device generator;
    std::uniform_real_distribution<double> distribution_roll(roll_ep0,
                                                             roll_ep1);
    std::uniform_real_distribution<double> distribution_pitch(pitch_ep0,
                                                              pitch_ep1);
    std::uniform_real_distribution<double> distribution_yaw(yaw_ep0, yaw_ep1);
    std::uniform_real_distribution<double> distribution_x(t0_ep0, t0_ep1);
    std::uniform_real_distribution<double> distribution_y(t1_ep0, t1_ep1);
    std::uniform_real_distribution<double> distribution_z(t2_ep0, t2_ep1);
    for (size_t i = 0; i < search_count; i++)
    {
        mutexval.lock();

        var[0] = distribution_roll(generator);
        var[1] = distribution_pitch(generator);
        var[2] = distribution_yaw(generator);
        var[3] = distribution_x(generator);
        var[4] = distribution_y(generator);
        var[5] = distribution_z(generator);
        mutexval.unlock();

        if (idx == "left")
        {
            double loss_new = CostFunction(var, idx, extrinsic_left_opt);
            if (loss_new < cur_left_loss)
            {
                lock_guard<std::mutex> lock(mutexleft);
                cur_left_loss = loss_new;
                extrinsic_left_opt =
                    extrinsic_left_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[0][i] += var[i];
                }
            }
        }
        if (idx == "right")
        {
            double loss_new = CostFunction(var, idx, extrinsic_right_opt);
            if (loss_new < cur_right_loss)
            {
                lock_guard<std::mutex> lock(mutexright);
                cur_right_loss = loss_new;
                extrinsic_right_opt =
                    extrinsic_right_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[1][i] += var[i];
                }
            }
        }
        if (idx == "behind")
        {
            double loss_new = CostFunction(var, idx, extrinsic_behind_opt);
            if (loss_new < cur_behind_loss)
            {
                lock_guard<std::mutex> lock(mutexbehind);
                cur_behind_loss = loss_new;
                extrinsic_behind_opt =
                    extrinsic_behind_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[2][i] += var[i];
                }
            }
        }
        if (idx == "front")
        {  // if fix back camera,front camera is calibrated at last
            double loss_new = CostFunction(var, idx, extrinsic_front_opt);
            if (loss_new < cur_front_loss)
            {
                lock_guard<std::mutex> lock(mutexfront);
                cur_front_loss = loss_new;
                extrinsic_front_opt =
                    extrinsic_front_opt * TransformUtil::GetDeltaT(var);
                for (int i = 0; i < 6; i++)
                {
                    bestVal_[2][i] += var[i];
                }
            }
        }
    }

    if (idx == "left")
    {
        lock_guard<std::mutex> lock(mutexleft);
        imgl_bev_gray =
            project_on_ground(imgl_gray, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_rgb =
            project_on_ground(imgl_rgb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);
        imgl_bev_atb =
            project_on_ground(imgl_atb, extrinsic_left_opt, intrinsic_left,
                              distortion_params_left, KG, brows, bcols, hl);

        imgl_bev_gray = tail(imgl_bev_gray, "l");
        imgl_bev_rgb  = tail(imgl_bev_rgb, "l");
        imgl_bev_atb  = tail(imgl_bev_atb, "l");
    }
    else if (idx == "right")
    {
        lock_guard<std::mutex> lock(mutexright);
        imgr_bev_gray =
            project_on_ground(imgr_gray, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_rgb =
            project_on_ground(imgr_rgb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);
        imgr_bev_atb =
            project_on_ground(imgr_atb, extrinsic_right_opt, intrinsic_right,
                              distortion_params_right, KG, brows, bcols, hr);

        imgr_bev_gray = tail(imgr_bev_gray, "r");
        imgr_bev_rgb  = tail(imgr_bev_rgb, "r");
        imgr_bev_atb  = tail(imgr_bev_atb, "r");
    }
    else if (idx == "behind")
    {
        lock_guard<std::mutex> lock(mutexbehind);
        imgb_bev_gray =
            project_on_ground(imgb_gray, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_rgb =
            project_on_ground(imgb_rgb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);
        imgb_bev_atb =
            project_on_ground(imgb_atb, extrinsic_behind_opt, intrinsic_behind,
                              distortion_params_behind, KG, brows, bcols, hb);

        imgb_bev_gray = tail(imgb_bev_gray, "b");
        imgb_bev_rgb  = tail(imgb_bev_rgb, "b");
        imgb_bev_atb  = tail(imgb_bev_atb, "b");
    }
    else
    {  // if fix back camera,front camera is calibrated at last
        lock_guard<std::mutex> lock(mutexfront);
        imgf_bev_gray =
            project_on_ground(imgf_gray, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hf);
        imgf_bev_rgb =
            project_on_ground(imgf_rgb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);
        imgf_bev_atb =
            project_on_ground(imgf_atb, extrinsic_front_opt, intrinsic_front,
                              distortion_params_front, KG, brows, bcols, hb);

        imgf_bev_gray = tail(imgf_bev_gray, "f");
        imgf_bev_rgb  = tail(imgf_bev_rgb, "f");
        imgf_bev_atb  = tail(imgf_bev_atb, "f");
    }
}