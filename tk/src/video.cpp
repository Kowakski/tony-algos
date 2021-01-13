#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include "comm.hpp"

using namespace cv;
using namespace std;

#include "framediff.hpp"

void show_help( void ){
    cout << "args err" <<endl;
    cout << "should use like this: "<<endl;
    cout << "./execu paht/video.mp4"<<endl;
}

/*
 *@ comment frm InputArray
 *@ 返回左上角和右下角
 */
void rect_area( InputArray frm, Point& lu, Point& rb ){
    uint x1, x2, y1,y2;
    if( frm.isMat() ){  //Mat type
        Mat src = frm.getMat();;
        x1 = src.rows;
        y1 = src.cols;

        x2 = 0;
        y2 = 0;

        cout << src.depth() << " /" << src.type() << endl;

        for( uint i = 0; i < src.rows; i++ ){
            for( uint j = 0; j < src.cols; j++ ){
                // cout << (int)src.at<uchar>(i,j) <<" ";
                if( (int)src.at<uchar>(i,j) > 20 ){
                    // cout << "更新坐标 "<<i<<" "<<j<<" "<<endl;
                    if( i > x2 ) x2 = i;
                    if( i < x1 ) x1 = i;
                    if( j > y2 ) y2 = j;
                    if( j < y1 ) y1 = j;
                }
            }
            // cout << endl;
        }
        // cout << endl;

        lu.x = y1;
        lu.y = x1;

        rb.x = y2;
        rb.y = x2;
    }

    return;
}

int main( int argc, char*argv[] ){
    if( argc < 2 ){
        show_help();
        return 0;
    }
    Mat frame;
    Mat frmdiff1, frmdiff2;
    uint frameCnt;
    string path = argv[1];
    cout << path <<endl;
    VideoCapture cap;

    if( !cap.open(path) ){
        cout << "open " << path << " fail" << endl;
        return 0;
    }

    FrameDiff fdiff(10);

    while( 1 ){
        cap >> frame;
        if( frame.empty() ){
            break;
        }

        resize( frame, frame, Size(WIDTH, HEIGHT), INTER_LINEAR );
        fdiff.fd( frame, frmdiff2 );
        if( !frmdiff2.empty() ){
            Point lu, rb;
            rect_area( frmdiff2, lu, rb );
            rectangle( frmdiff2, lu, rb, Scalar(255,0,0) );
            cout << "lu " <<lu<<" rb "<< rb << endl;
            printf("show result\n");
            imshow( "VIDEO", frmdiff2 );
            // cout << frmdiff2 << endl;
        }
        waitKey(10);
    }

    return 1;
}