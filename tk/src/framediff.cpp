#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include "framediff.hpp"
#include "debug.hpp"

FrameDiff::FrameDiff( uint spsize ){
    spsz = spsize;
    queue.clear();
    return;
}


/*
 *@comment frame difference
 */
void FrameDiff::fd( InputArray src1, OutputArray dst ){
    Mat src1gray, src2gray;
    cvtColor( src1, src1gray, COLOR_BGR2GRAY, 1 );

    if( queue.size() >= spsz ){
        queue.erase( queue.begin() );   //queue too long ,delete the eraliest one
    }

    queue.push_back( src1gray );

    if( queue.size() < spsz ){
        // dst.clear(); //不能用这个，只有新分配的内存可以用这个，会释放掉内存
        dbg("FrameDiff queue is short\n");
        return;
    }
    InputArray src_(queue[0]);

    // imshow("PREVIOUS",src_);
    // imshow("CUR", src1gray);

    // subtract( src_, src1gray, dst );
    absdiff( src_, src1gray, dst );
    threshold( dst, dst, 50, 255, THRESH_BINARY );
    dilate( dst, dst, Mat(), Point(-1,-1), 2 );
    erode( dst, dst, Mat(), Point(-1,-1), 6 );
    return;
}