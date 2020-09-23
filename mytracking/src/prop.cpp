#include "Eigen/Dense"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;
using namespace Eigen;

int main( int argc, char* argv[] ){
    #if 0
    cv::Mat img1, img2;
    // img.create(3,10, CV_16UC3 );
    // img.setTo( cv::Scalar(1.0f, 2.0f, 3.0f) );
    cout << "enter"<<endl;
    img1 = imread("./003.png", IMREAD_GRAYSCALE  );
    cv::resize( img1, img1, Size( round(img1.rows/3), round( img1.cols/2 ) ), INTER_LINEAR );
    if( img1.empty() ){
        cout << "read err!!"<<endl;
        exit(1);
    }
    MatrixXd m( img1.rows, img1.cols );
    cv2eigen(img1,m);

    cout << "read img 1" <<endl;
    img2 = imread("./004.png", IMREAD_GRAYSCALE );
    cv::resize( img2, img2, img1.size(), INTER_LINEAR  );
    if( img2.empty() ){
     cout << "read err !!"<< endl;
     exit(1);
    }
    cout << "read img 2" << endl;
    namedWindow("img1");
    imshow("img1", img1);
    cout << "show img 1"  << endl;
    namedWindow("img2");
    imshow("img2", img2);
    cout << "show img 2" << endl;

    cv::Mat img3 = img2 - img1;
    size_t i = 0;
    for( auto it = img3.begin<uchar>(); it != img3.end<uchar>(); it++ ){
     if( (i%30 == 0)&&( i/img3.cols >= img3.rows/3 )&&( i/img3.cols <= img3.rows*2/3 ) ){
      *it = 255;
     }
     i++;
    }
    namedWindow("img3");
    imshow("img3", img3);
    cout << "show img 3" << endl;
    cout << img1.type() << " "<<img1.depth() <<" //"<<img2.type()<<" "<<img2.depth() << " //"<<img3.type() << " "<<img3.depth() <<endl;
    #endif

    #if 0
    cout << img.size() << endl;
    cout << img.depth() <<endl;
    cout << "step " << img.step[0] << " "<<img.step[1] << endl;
    cout << img.type() << endl;
    // cout << img.at<float>(2,2) << " " << img.at<float>(3,3) << endl;
    cout << img.at<Vec3f>(2,2)[0]<<" "<<img.at<Vec3f>(2,2)[1]<<" "<<img.at<Vec3f>(2,2)[2]<<endl;
    // cout << img(2,2)[0] << img(2,2)[1]<<endl;
    cout << img.ptr<unsigned short>(1)[0] << img.ptr<unsigned short>(1)[1] << img.ptr<unsigned short>(1)[2] << endl;

    cv::Mat image(10 , 10 , CV_8UC1, cv::Scalar(1) );

    for (auto it = image.begin<uchar>(); it != image.end<uchar>(); ++it){
        std::cout << int((*it)) << " ";
    }
#endif
    #if 1
    cv::Mat image3c(10, 10, CV_8UC3, cv::Scalar(1,2,3) );

    for (auto it = image3c.begin<cv::Vec3b>(); it != image3c.end<cv::Vec3b>(); ++it){
        std::cout << int((*it)[0]) << " " << int((*it)[1]) << " " << int((*it)[2])<<"  ";
    }

    std::cout << std::endl;
    std::cout << image3c.isContinuous()<<std::endl;

    std::cout << "乘以3之后："<<std::endl;
    cv::Mat image3c3 = image3c*3;
    for( auto it = image3c3.begin<cv::Vec3b>(); it != image3c3.end<cv::Vec3b>(); ++it ){
     std::cout << int((*it)[0]) << " " << int((*it)[1]) << " " << int((*it)[2])<<"  ";
    }
    std::cout << std::endl;
   #endif
    while( (char)waitKey(10) != 'q' ){}

    return 0;
}