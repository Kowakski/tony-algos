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

void show_help( void ){
    cout << "args err" <<endl;
    cout << "should use like this: "<<endl;
    cout << "./execu paht/video.mp4"<<endl;
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

    frameCnt = 0;
    cap >> frmdiff1;
    resize( frmdiff1, frmdiff1, Size(WIDTH, HEIGHT), INTER_LINEAR );

    while( 1 ){
        cap >> frame;
        if( frame.empty() ){
            break;
        }
        frameCnt++;

        resize( frame, frame, Size(480,360), INTER_LINEAR );

        if( frameCnt == 5 ){
            frameCnt = 0;

        }

        imshow( "VIDEO", frame );

        waitKey(10);
    }

    return 1;
}