// #include< opencv2/opencv.hpp >
#include <opencv2/opencv.hpp>
#include <iostream>

int main( int argc, char** argv ){
    cv::Mat img = cv::imread( "dog1.jpeg" );
    if( img.empty() ){
        std::cout << "img not exist" << std::endl;
    }
    int rows = img.rows;
    int colums = img.cols;
    std::cout << img.size <<" / "<< img.size() << " / "<< img.type() << " / "<< img.channels() <<std::endl;

    for( int i = 0; i < rows; i++ ){
        for( int j = 0; j < colums; j++ ){
    std::cout << "("<< (unsigned int)img.at<cv::Vec3b>(i,j)[0] <<","<<(unsigned int)img.at<cv::Vec3b>(i,j)[1] <<"," << (unsigned int)img.at<cv::Vec3b>(i,j)[2] <<")" << " ";

        }
        std::cout<<std::endl;
    }
    // cv::namedWindow( "old man", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "old man", img );
    // cv::waitKey( 0 );
    // cv::destroyWindow("old man");
    return 0;
}