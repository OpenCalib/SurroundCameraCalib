#include"optimizer.h"
#include "transform_util.h"
#include<fstream>

cv::Mat Optimizer::eigen2mat(Eigen::MatrixXd A) {
    cv::Mat B;
    cv::eigen2cv(A, B);

    return B;
}

Mat Optimizer::gray_gamma(Mat img){
	Mat gray;
	cvtColor(img,gray,COLOR_BGR2GRAY);
    double contrast = 1.1;
    double brightness = 0;
    double delta = 30 ;
	for(int i=0;i<img.rows;i++){
	    for(int j=0;j<img.cols;j++){
		    int g = gray.at< uchar>(i,j);
		    gray.at< uchar>(i,j) = saturate_cast<uchar>(contrast * (gray.at<uchar>(i,j)-delta) + brightness);
        }
	}
	return gray;
}

double Optimizer::getPixelValue (Mat *image, float x, float y ){
    // 法1：双线性插值
    uchar* data = & image->data[ int ( y ) * image->step + int ( x ) ];
    float xx = x - floor ( x );
    float yy = y - floor ( y );
    return float (
               ( 1-xx ) * ( 1-yy ) * data[0] +
               xx* ( 1-yy ) * data[1] +
               ( 1-xx ) *yy*data[ image->step ] +
               xx*yy*data[image->step+1]
           );
}

vector<Point> Optimizer::readfromcsv(string filename){
	vector<Point> pixels;
	ifstream inFile(filename, ios::in);
	string lineStr;
	while(getline(inFile,lineStr)){
		Point pixel;
		istringstream record(lineStr);
		string x,y;
		record>>x;
		pixel.x=atoi(x.c_str());
		record>>y;
		pixel.y=atoi(y.c_str());
		pixels.push_back(pixel);
	}
	return pixels;
}

Mat Optimizer::tail(Mat img,string index){
    if(index=="f"){
        cv::Rect m_select_f=Rect(0,0,img.cols,450);
	    Mat cropped_image_f = img(m_select_f);
        Mat border(img.rows-450,img.cols,cropped_image_f.type(),Scalar(0,0,0));
        Mat dst_front;
        vconcat(cropped_image_f,border,dst_front);
        return dst_front;     
    }else if(index=="l"){
        cv::Rect m_select_l=Rect(0,0,450,img.rows);
	    Mat cropped_image_l = img(m_select_l);
        Mat border2(img.rows,img.cols-450,cropped_image_l.type(),Scalar(0,0,0));
        Mat dst_left;
        hconcat(cropped_image_l,border2,dst_left);
        return dst_left;
    }else if(index=="b"){
       cv::Rect m_select_b=Rect(0,img.rows-400,img.cols,400);
	    Mat cropped_image_b = img(m_select_b);
        Mat border1(img.rows-400,img.cols,cropped_image_b.type(),Scalar(0,0,0));
        Mat dst_behind;
        vconcat(border1,cropped_image_b,dst_behind);  
        return dst_behind;
    }else if(index=="r"){
        cv::Rect m_select_r=Rect(img.cols-450,0,450,img.rows);
	    Mat cropped_image_r = img(m_select_r);
        Mat border3(img.rows,img.cols-450,cropped_image_r.type(),Scalar(0,0,0));
        Mat dst_right;
        hconcat(border3,cropped_image_r,dst_right);
        return dst_right;
    }
    return Mat(img.rows,img.cols,img.type());
}

void Optimizer::SaveOptResult(const string filename){
	Mat opt_after=generate_surround_view(imgf_bev_rgb,imgl_bev_rgb,imgb_bev_rgb,imgr_bev_rgb);
    imwrite(filename,opt_after);
	imshow("after_all_cameras_calib",opt_after);
	waitKey(0);
}

void Optimizer::show(string idx,string filename){
	Mat dst,dst1;
	
	if(idx=="right"){//first
		imgr_bev_rgb=project_on_ground(imgr_rgb,extrinsic_right_opt,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
        imgr_bev_rgb=tail(imgr_bev_rgb,"r");
		addWeighted(imgf_bev_rgb,0.5,imgr_bev_rgb,0.5,3,dst);
		imshow("after_right_calib",dst);
		waitKey(0);
		imwrite(filename,dst);
	}
	if(idx=="behind"){//second
		imgb_bev_rgb=project_on_ground(imgb_rgb,extrinsic_behind_opt,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hb);
        imgb_bev_rgb=tail(imgb_bev_rgb,"b");
		addWeighted(imgb_bev_rgb,0.5,imgr_bev_rgb,0.5,3,dst);
		addWeighted(dst,1,imgl_bev_rgb,0.5,3,dst1);
		imshow("after_behind_calib",dst1);
		waitKey(0);
		imwrite(filename,dst1);
	}
	if(idx=="left"){//third
		imgl_bev_rgb=project_on_ground(imgl_rgb,extrinsic_left_opt,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
        imgl_bev_rgb=tail(imgl_bev_rgb,"l");
		addWeighted(imgl_bev_rgb,0.5,imgf_bev_rgb,0.5,3,dst1);
		imshow("after_left_calib",dst1);
		waitKey(0);
		imwrite(filename,dst1);
	}
}

void Optimizer::world2cam(double point2D[2], double point3D[3],Eigen::Matrix3d K,vector<double> D){
        double norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
        double theta = atan(point3D[2] / norm);
        double t, t_i;
        double rho, x, y;
        double invnorm;
        int i;

        if (norm != 0){
            invnorm = 1 / norm;
            t = theta;
            rho = D[0];
            t_i = 1;

            for (i = 1; i <D.size(); i++){
                t_i *= t;
                rho += t_i * D[i];
            }

            x = point3D[0] * invnorm * rho;
            y = point3D[1] * invnorm * rho;

            point2D[0] = x * K(0,0) + y * K(0, 1) + K(0, 2);
            point2D[1] = x * K(1, 0) + y + K(1, 2);
        }else{
            point2D[0] = K(0, 2);
            point2D[1] = K(1, 2);
        }
}

void Optimizer::distortPointsOcam(cv::Mat &P_GC1,cv::Mat &p_GC,Eigen::Matrix3d &K_C,vector<double> &D_C){
    double M[3];
    double m[2];
    for (int i = 0; i < P_GC1.cols; i++) {
      M[0] = P_GC1.at<cv::Vec2d>(0, i)[0];
      M[1] = P_GC1.at<cv::Vec2d>(0, i)[1];
      M[2] = -1;
      world2cam(m, M,K_C,D_C);
      p_GC.at<cv::Vec2d>(0, i)[0] = m[0];
      p_GC.at<cv::Vec2d>(0, i)[1] = m[1];
    }
}

void Optimizer::distortPoints(cv::Mat &P_GC1,cv::Mat &p_GC,Eigen::Matrix3d &K_C){
    for(int i=0;i<P_GC1.cols;i++){
		double x=P_GC1.at<cv::Vec2d>(0,i)[0];
		double y=P_GC1.at<cv::Vec2d>(0,i)[1];
		
		double u=x*K_C(0,0)+K_C(0,2);
		double v=y*K_C(1,1)+K_C(1,2);
	
		p_GC.at<cv::Vec2d>(0,i)[0]=u;
		p_GC.at<cv::Vec2d>(0,i)[1]=v;
	}
}

void Optimizer::initializeK() {
	Eigen::Matrix3d K_F;
	Eigen::Matrix3d K_L;
	Eigen::Matrix3d K_B;
	Eigen::Matrix3d K_R;

    //carla
   	K_F<<390.425287,  0.00000000, 750,
		0.00000000, 390.425287,   750,
		0.00000000, 0.00000000, 1.00000000;
	K_L<<390.425287,  0.00000000, 750,
		0.00000000, 390.425287,   750,
		0.00000000, 0.00000000, 1.00000000;
	K_B<<390.425287,  0.00000000, 750,
		0.00000000, 390.425287,   750,
		0.00000000, 0.00000000, 1.00000000;
	K_R<<390.425287,  0.00000000, 750,
		0.00000000, 390.425287,   750,
		0.00000000, 0.00000000, 1.00000000; 


	intrinsic_front=K_F;
	intrinsic_left=K_L;
	intrinsic_behind=K_B;
	intrinsic_right=K_R;
    return;
}

void Optimizer::initializeD() {
	vector<double> D_F;
	vector<double> D_L;
	vector<double> D_B; 
	vector<double> D_R;

    //carla-pinhole
    D_F={0,0,0,0};
    D_L={0,0,0,0};
    D_B={0,0,0,0};
    D_R={0,0,0,0};

    distortion_params_front=D_F;
	distortion_params_left=D_L;
	distortion_params_behind=D_B;
	distortion_params_right=D_R;

}

void Optimizer::initializePose() {//ground->camera
	Eigen::Matrix4d T_FG;
	Eigen::Matrix4d T_LG;
    Eigen::Matrix4d T_BG;
	Eigen::Matrix4d T_RG;

    //carla
    T_FG<< 1,    0,   0,    0,
   		   0,    0,   1, -4.1,
  		   0,   -1,   0, -2.5,
  		   0,    0,   0,    1;
    T_LG<< 0,   -1,    0,    0,
  		   0,    0,    1,  -4.1,
  		  -1,    0,    0,   -1,
  		   0,    0,    0,    1;
    T_BG<<-1,    0,    0,    0,
   		   0,    0,    1, -4.1,
   		   0,    1,    0,   -2,
   		   0,    0,    0,    1;
    T_RG<< 0,    1,    0,    0,
  		   0,    0,    1, -4.1,
  		   1,    0,    0,   -1,
  		   0,    0,    0,    1;

    // left disturbance
    Eigen::Matrix4d left_disturbance;
	Eigen::Matrix3d left_disturbance_rot_mat;
	Vec3f left_disturbance_rot_euler;//R(euler)
	Mat_<double> left_disturbance_t=(Mat_<double>(3, 1)<<0.09,0.070,0.085);
	left_disturbance_rot_euler<<2.9,2.6,2.8;
	left_disturbance_rot_mat=TransformUtil::eulerAnglesToRotationMatrix(left_disturbance_rot_euler);
	left_disturbance=TransformUtil::R_T2RT(TransformUtil::eigen2mat(left_disturbance_rot_mat),left_disturbance_t);
    T_LG*=left_disturbance;

    // rightdisturbance
    Eigen::Matrix4d right_disturbance;
	Eigen::Matrix3d right_disturbance_rot_mat;
	Vec3f right_disturbance_rot_euler;
	Mat_<double> right_disturbance_t=(Mat_<double>(3, 1)<<0.06,0.09,0.10);
	right_disturbance_rot_euler<<2.9,-2.2,2.85;
	right_disturbance_rot_mat=TransformUtil::eulerAnglesToRotationMatrix(right_disturbance_rot_euler);
	right_disturbance=TransformUtil::R_T2RT(TransformUtil::eigen2mat(right_disturbance_rot_mat),right_disturbance_t);
    T_RG*=right_disturbance;

    //behind disturbance
    Eigen::Matrix4d behind_disturbance;
	Eigen::Matrix3d behind_disturbance_rot_mat;
	Vec3f behind_disturbance_rot_euler;
	Mat_<double> behind_disturbance_t=(Mat_<double>(3, 1)<<0.095,0.099,0.079);
	behind_disturbance_rot_euler<<1.0,2.86,2.70;
	behind_disturbance_rot_mat=TransformUtil::eulerAnglesToRotationMatrix(behind_disturbance_rot_euler);
	behind_disturbance=TransformUtil::R_T2RT(TransformUtil::eigen2mat(behind_disturbance_rot_mat),behind_disturbance_t);
    T_BG*=behind_disturbance;

	extrinsic_front=T_FG;
	extrinsic_left=T_LG;
	extrinsic_behind=T_BG;
	extrinsic_right=T_RG;

    cout<<"extrinsic_front:"<<endl<<extrinsic_front<<endl;
    cout<<"eular:"<<endl<<TransformUtil::Rotation2Eul(extrinsic_front.block(0,0,3,3))<<endl;
    cout<<"extrinsic_left:"<<endl<<extrinsic_left<<endl;
    cout<<"eular:"<<TransformUtil::Rotation2Eul(extrinsic_left.block(0,0,3,3))<<endl;
    cout<<"extrinsic_right:"<<endl<<extrinsic_right<<endl;
    cout<<"eular:"<<TransformUtil::Rotation2Eul(extrinsic_right.block(0,0,3,3))<<endl;
    cout<<"extrinsic_behind:"<<endl<<extrinsic_behind<<endl;
    cout<<"eular:"<<TransformUtil::Rotation2Eul(extrinsic_behind.block(0,0,3,3))<<endl;
    return;
}

void Optimizer::initializeKG(){
    Eigen::Matrix3d K_G;
    //carla
    K_G<<390.425287,  0.00000000, 750,
	0.00000000, 390.425287,   750,
	0.00000000, 0.00000000, 1.00000000;
    KG=K_G;
}

void Optimizer::initializeHeight(){
    hf=5.1;
    hl=5.1;
    hb=5.1;
    hr=5.1;
}

Optimizer::Optimizer(const Mat *imgf,const Mat *imgl,const Mat *imgb,const Mat *imgr,int camera_model_index,int rows,int cols){
	imgf_rgb = *imgf;
	imgf_gray=gray_gamma(imgf_rgb);

    imgl_rgb = *imgl;  
	imgl_gray=gray_gamma(imgl_rgb);
	
	imgb_rgb = *imgb;
	imgb_gray=gray_gamma(imgb_rgb);

	imgr_rgb = *imgr;
	imgr_gray=gray_gamma(imgr_rgb);

	initializeK();
	initializeD();
	initializePose();
    initializeKG();
    initializeHeight();

    camera_model=camera_model_index;

    brows=rows;
    bcols=cols;

    bestVal_.resize(3,vector<double>(6));

	imgf_bev=project_on_ground(imgf_gray,extrinsic_front,intrinsic_front,distortion_params_front,KG,brows,bcols,hf);
	imgf_bev_rgb=project_on_ground(imgf_rgb,extrinsic_front,intrinsic_front,distortion_params_front,KG,brows,bcols,hf);
    imgf_bev=tail(imgf_bev,"f");
    imgf_bev_rgb=tail(imgf_bev_rgb,"f");
 
}

Optimizer::~Optimizer() {}

cv::Mat Optimizer::project_on_ground(cv::Mat img, Eigen::Matrix4d T_CG,
						  Eigen::Matrix3d K_C,vector<double> D_C,
						  Eigen::Matrix3d K_G,int rows, int cols,float height){
	cv::Mat p_G = cv::Mat::ones(3,rows*cols,CV_64FC1);
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			p_G.at<double>(0,cols*i+j) = j;
			p_G.at<double>(1,cols*i+j) = i;
		}
	}
	
	cv::Mat P_G = cv::Mat::ones(4,rows*cols,CV_64FC1);
	P_G(cv::Rect(0,0,rows*cols,3)) = eigen2mat(K_G.inverse())*p_G*height;
    // P_G(cv::Rect(0,2,rows*cols,1)) = 0;
	cv::Mat P_GC = cv::Mat::zeros(4,rows*cols,CV_64FC1);
    cv::Mat T_CG_=(cv::Mat_<double>(4,4)<<T_CG(0,0),T_CG(0,1),T_CG(0,2),T_CG(0,3),
                               T_CG(1,0),T_CG(1,1),T_CG(1,2),T_CG(1,3),
                               T_CG(2,0),T_CG(2,1),T_CG(2,2),T_CG(2,3),
                               T_CG(3,0),T_CG(3,1),T_CG(3,2),T_CG(3,3));
	P_GC= T_CG_*P_G;

	cv::Mat P_GC1 = cv::Mat::zeros(1,rows*cols,CV_64FC2);
	vector<cv::Mat> channels(2);
	cv::split(P_GC1, channels);
	channels[0] = P_GC(cv::Rect(0,0,rows*cols,1))/P_GC(cv::Rect(0,2,rows*cols,1));
	channels[1] = P_GC(cv::Rect(0,1,rows*cols,1))/P_GC(cv::Rect(0,2,rows*cols,1));
	cv::merge(channels, P_GC1);

	cv::Mat p_GC = cv::Mat::zeros(1,rows*cols,CV_64FC2);
    cv::Mat K_C_=(cv::Mat_<double>(3,3)<<K_C(0,0),K_C(0,1),K_C(0,2),
                                       K_C(1,0),K_C(1,1),K_C(1,2),
                                       K_C(2,0),K_C(2,1),K_C(2,2));

	if(camera_model==0){
        cv::fisheye::distortPoints(P_GC1,p_GC,K_C_,D_C);   //fisheye
    }
    else if(camera_model==1){
        distortPointsOcam(P_GC1,p_GC,K_C,D_C);             //Ocam
    }else{
        distortPoints(P_GC1,p_GC,K_C);                     //pinhole
    }
  
	p_GC.reshape(rows,cols);
	cv::Mat p_GC_table = p_GC.reshape(0,rows);
	cv::Mat p_GC_table_32F;
	p_GC_table.convertTo(p_GC_table_32F,CV_32FC2);
	
	cv::Mat img_GC;
	cv::remap(img,img_GC,p_GC_table_32F,cv::Mat(),cv::INTER_LINEAR);
	return img_GC;
}

Mat Optimizer::generate_surround_view(Mat img_GF, Mat img_GL,Mat img_GB, Mat img_GR){
	Mat dst1,dst2,dst3;
	addWeighted(img_GF,0.5,img_GL,0.5,3,dst1);
	addWeighted(dst1,1.0,img_GB,0.5,3,dst2);
	addWeighted(dst2,1.0,img_GR,0.5,3,dst3);
	return dst3;
}

void Optimizer::Calibrate_left(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    vector<double> var(6, 0);
    string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};
	
	max_left_loss=cur_left_loss = CostFunction(var,"left","less");

    random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "left","less");

}

void Optimizer::Calibrate_right(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    vector<double> var(6, 0);
    string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};

	max_right_loss=cur_right_loss = CostFunction(var,"right","less");

    random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "right","less");


}

void Optimizer::Calibrate_behind(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    vector<double> var(6, 0);
    string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};
	
 
	max_behind_loss=cur_behind_loss = CostFunction(var,"behind","less");
 
    random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "behind","less");

}

void Optimizer::fine_Calibrate_left(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    fine_random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "left","less");
}

void Optimizer::fine_Calibrate_right(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    fine_random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "right","less");
}

void Optimizer::fine_Calibrate_behind(int search_count,double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                            double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1) {
    fine_random_search_params(search_count,
        roll_ep0,roll_ep1,pitch_ep0,pitch_ep1,yaw_ep0,yaw_ep1,
        t0_ep0,t0_ep1,t1_ep0,t1_ep1,t2_ep0,t2_ep1,
        "behind","less");
}

double Optimizer::CostFunction(const vector<double> var,string idx,string model) {
	if(idx=="right"){
		Eigen::Matrix4d Tr = extrinsic_right;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tr*=deltaT;
		double loss=back_camera_and_compute_loss(imgf_bev,imgr_gray,Tr,"fr",model,hr);
		return loss;
	}else if(idx=="left"){
		Eigen::Matrix4d Tl = extrinsic_left;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tl*=deltaT;
		double loss=back_camera_and_compute_loss(imgf_bev,imgl_gray,Tl,"fl",model,hl);
		return loss;
	}else{
		Eigen::Matrix4d Tb = extrinsic_behind;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tb*=deltaT;
		double loss=back_camera_and_compute_loss(imgl_bev,imgb_gray,Tb,"bl",model,hb);
		loss+=back_camera_and_compute_loss(imgr_bev,imgb_gray,Tb,"br",model,hb);
		return loss;
	}	
}

double Optimizer::fine_CostFunction(const vector<double> var,string idx,string model) {
	if(idx=="right"){
		Eigen::Matrix4d Tr = extrinsic_right_opt;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tr*=deltaT;
		double loss=back_camera_and_compute_loss(imgf_bev,imgr_gray,Tr,"fr",model,hr);
		return loss;
	}else if(idx=="left"){
		Eigen::Matrix4d Tl = extrinsic_left_opt;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tl*=deltaT;
		double loss=back_camera_and_compute_loss(imgf_bev,imgl_gray,Tl,"fl",model,hl);
		return loss;
	}else{
		Eigen::Matrix4d Tb = extrinsic_behind_opt;
		Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);	
		Tb*=deltaT;
		double loss=back_camera_and_compute_loss(imgl_bev,imgb_gray,Tb,"bl",model,hb);
		loss+=back_camera_and_compute_loss(imgr_bev,imgb_gray,Tb,"br",model,hb);
		return loss;
	}	
}

double Optimizer::back_camera_and_compute_loss(Mat img1,Mat img2,Eigen::Matrix4d T,string idx,string model,double height){
	vector<Point> pixels;
    Eigen::Matrix3d KC;
    vector<double>DC;
    Mat pG;
    Mat PG;
    Mat show1;
	if(idx=="fl"){
        // show1=imgl_rgb.clone();
        DC=distortion_params_left;
        KC=intrinsic_left;
        pG=pG_fl;
        PG=PG_fl;
        pixels=fl_pixels_texture_less;
	}else if(idx=="fr"){
        // show1=imgr_rgb.clone();
        DC=distortion_params_right;
        KC=intrinsic_right;
        pG=pG_fr;
        PG=PG_fr;
        pixels=fr_pixels_texture_less;
	}else if(idx=="bl"){
        // show1=imgb_rgb.clone();
        DC=distortion_params_behind;
        KC=intrinsic_behind;
        pG=pG_bl;
        PG=PG_bl;
        pixels=bl_pixels_texture_less;
	}else{
        // show1=imgb_rgb.clone();
        DC=distortion_params_behind;
        KC=intrinsic_behind;
        pG=pG_br;
        PG=PG_br;
        pixels=br_pixels_texture_less;
	}
	double loss=0;
    int failcount=0;
    int size=pixels.size();
    cv::Mat PG2C = cv::Mat::zeros(4,size,CV_64FC1);
    PG2C=eigen2mat(T)*PG;
    Mat PG2C1 = cv::Mat::zeros(1,size,CV_64FC2);
	vector<cv::Mat> channels(2);
	cv::split(PG2C1, channels);
	channels[0] = PG2C(cv::Rect(0,0,size,1))/PG2C(cv::Rect(0,2,size,1));
	channels[1] = PG2C(cv::Rect(0,1,size,1))/PG2C(cv::Rect(0,2,size,1));

	cv::merge(channels, PG2C1);
    Mat pG2C(1,size,CV_64FC2);
    if(camera_model==0)
        distortPointsOcam(PG2C1,pG2C,KC,DC);   
    else if(camera_model==1)
        cv::fisheye::distortPoints(PG2C1,pG2C,eigen2mat(KC),DC);
    else
        distortPoints(PG2C1,pG2C,KC);
	for(int i=0;i<size;i++){
        double x=pG.at<double>(0, i);
        double y=pG.at<double>(1, i);         
        double x1=pG2C.at<Vec2d>(0, i)[0];
        double y1=pG2C.at<Vec2d>(0, i)[1];        
	    if(x1>0&&y1>0&&x1<img2.cols&&y1<img2.rows){
	    	loss+=fabs(getPixelValue(&img1,x,y)-getPixelValue(&img2,x1,y1));
	    }else{
	    	failcount++;
	    	if(failcount>30)	
	    		return INT_MAX;
	    }
    }
 
	return loss;
}

void Optimizer::random_search_params(int search_count, 
									double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                                    double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1,
                                    string idx,string model){
	vector<double> var(6, 0.0);
    double resolution_r=100;
    double resolution_t=50;

    random_device generator;
	std::uniform_int_distribution<int> distribution_roll(roll_ep0*resolution_r,roll_ep1*resolution_r);
	std::uniform_int_distribution<int> distribution_pitch(pitch_ep0*resolution_r,pitch_ep1*resolution_r);
	std::uniform_int_distribution<int> distribution_yaw(yaw_ep0*resolution_r,yaw_ep1*resolution_r);	
	std::uniform_int_distribution<int> distribution_x(t0_ep0*resolution_t, t0_ep1*resolution_t);
	std::uniform_int_distribution<int> distribution_y(t1_ep0*resolution_t, t1_ep1*resolution_t);
	std::uniform_int_distribution<int> distribution_z(t2_ep0*resolution_t, t2_ep1*resolution_t);

	for (size_t i = 0; i < search_count; i++) {
        mutexval.lock();
		var[0] = double(distribution_roll(generator) )/resolution_r;
		var[1] = double(distribution_pitch(generator))/resolution_r; 
		var[2] = double(distribution_yaw(generator)  )/resolution_r;
		var[3] = double(distribution_x(generator))/resolution_t;
		var[4] = double(distribution_y(generator))/resolution_t;
		var[5] = double(distribution_z(generator))/resolution_t;

        mutexval.unlock();
 
		double loss_new = CostFunction(var,idx,model);
		if(idx=="left"&&loss_new<cur_left_loss){
            lock_guard<std::mutex> lock(mutexleft);
			cur_left_loss=loss_new;
			extrinsic_left_opt = extrinsic_left * TransformUtil::GetDeltaT(var);	
			bestVal_[0]=var;
		}
		if(idx=="right"&&loss_new<cur_right_loss){
            lock_guard<std::mutex> lock(mutexright);
			cur_right_loss=loss_new;
			extrinsic_right_opt = extrinsic_right * TransformUtil::GetDeltaT(var);
			bestVal_[1]=var;
		}
		if(idx=="behind"&&loss_new<cur_behind_loss){
            lock_guard<std::mutex> lock(mutexbehind);
			cur_behind_loss=loss_new;
			extrinsic_behind_opt = extrinsic_behind * TransformUtil::GetDeltaT(var);
			bestVal_[2]=var;
		}		
	}

	if(idx=="left"){
		imgl_bev=project_on_ground(imgl_gray,extrinsic_left,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
		imgl_bev_rgb=project_on_ground(imgl_rgb,extrinsic_left,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
        imgl_bev=tail(imgl_bev,"l");
        imgl_bev_rgb=tail(imgl_bev_rgb,"l");
	}else if(idx=="right"){
		imgr_bev=project_on_ground(imgr_gray,extrinsic_right,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
		imgr_bev_rgb=project_on_ground(imgr_rgb,extrinsic_right,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
        imgr_bev=tail(imgr_bev,"r");
        imgr_bev_rgb=tail(imgr_bev_rgb,"r");
	}else{
        imgb_bev=project_on_ground(imgb_gray,extrinsic_behind,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hb);
        imgb_bev_rgb=project_on_ground(imgb_rgb,extrinsic_behind,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hb);
        imgb_bev=tail(imgb_bev,"b");
        imgb_bev_rgb=tail(imgb_bev_rgb,"b");
    }
}

void Optimizer::fine_random_search_params(int search_count, 
									double roll_ep0,double roll_ep1,double pitch_ep0,double pitch_ep1,double yaw_ep0,double yaw_ep1,
                                    double t0_ep0,double t0_ep1,double t1_ep0,double t1_ep1,double t2_ep0,double t2_ep1,
                                    string idx,string model){
	vector<double> var(6, 0.0);
    double resolution_r=20;
    double resolution_t=10;

    random_device generator;
	static std::uniform_int_distribution<int> distribution_roll(roll_ep0*resolution_r,roll_ep1*resolution_r);
	static std::uniform_int_distribution<int> distribution_pitch(pitch_ep0*resolution_r,pitch_ep1*resolution_r);
	static std::uniform_int_distribution<int> distribution_yaw(yaw_ep0*resolution_r,yaw_ep1*resolution_r);	
	static std::uniform_int_distribution<int> distribution_x(t0_ep0*resolution_t, t0_ep1*resolution_t);
	static std::uniform_int_distribution<int> distribution_y(t1_ep0*resolution_t, t1_ep1*resolution_t);
	static std::uniform_int_distribution<int> distribution_z(t2_ep0*resolution_t, t2_ep1*resolution_t);
	
	for (size_t i = 0; i < search_count; i++) {
        mutexval.lock();
		var[0] = double(distribution_roll(generator) )/resolution_r;
		var[1] = double(distribution_pitch(generator))/resolution_r; 
		var[2] = double(distribution_yaw(generator)  )/resolution_r;
		var[3] = double(distribution_x(generator))/resolution_t;
		var[4] = double(distribution_y(generator))/resolution_t;
		var[5] = double(distribution_z(generator))/resolution_t;
        mutexval.unlock();
		double loss_new = fine_CostFunction(var,idx,model);
		if(idx=="left"&&loss_new<cur_left_loss){
            lock_guard<std::mutex> lock(mutexleft);
			cur_left_loss=loss_new;
			extrinsic_left_opt = extrinsic_left_opt * TransformUtil::GetDeltaT(var);	
			bestVal_[0]=var;
		}
		if(idx=="right"&&loss_new<cur_right_loss){
            lock_guard<std::mutex> lock(mutexright);
			cur_right_loss=loss_new;
			extrinsic_right_opt = extrinsic_right_opt * TransformUtil::GetDeltaT(var);
			bestVal_[1]=var;
		}
		if(idx=="behind"&&loss_new<cur_behind_loss){
            lock_guard<std::mutex> lock(mutexbehind);
			cur_behind_loss=loss_new;
			extrinsic_behind_opt = extrinsic_behind_opt * TransformUtil::GetDeltaT(var);
			bestVal_[2]=var;
		}		
	}


	if(idx=="left"){
		imgl_bev=project_on_ground(imgl_gray,extrinsic_left,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
		imgl_bev_rgb=project_on_ground(imgl_rgb,extrinsic_left,intrinsic_left,distortion_params_left,KG,brows,bcols,hl);
        imgl_bev=tail(imgl_bev,"l");
        imgl_bev_rgb=tail(imgl_bev_rgb,"l");
	}else if(idx=="right"){
		imgr_bev=project_on_ground(imgr_gray,extrinsic_right,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
		imgr_bev_rgb=project_on_ground(imgr_rgb,extrinsic_right,intrinsic_right,distortion_params_right,KG,brows,bcols,hr);
        imgr_bev=tail(imgr_bev,"r");
        imgr_bev_rgb=tail(imgr_bev_rgb,"r");
	}else{
        imgb_bev=project_on_ground(imgb_gray,extrinsic_behind,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hr);
        imgb_bev_rgb=project_on_ground(imgb_rgb,extrinsic_behind,intrinsic_behind,distortion_params_behind,KG,brows,bcols,hr);
        imgb_bev=tail(imgb_bev,"b");
        imgb_bev_rgb=tail(imgb_bev_rgb,"b");
    }
}