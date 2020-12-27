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

    FrameDiff fdiff(5);

    while( 1 ){
        cap >> frame;
        if( frame.empty() ){
            break;
        }

        resize( frame, frame, Size(WIDTH, HEIGHT), INTER_LINEAR );
        fdiff.fd( frame, frmdiff2 );
        if( !frmdiff2.empty() ){
            printf("show result\n");
            imshow( "VIDEO", frmdiff2 );
            cout << frmdiff2 << endl;
        }
        waitKey(10);
    }

    return 1;
}