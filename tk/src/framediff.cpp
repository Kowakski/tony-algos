#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace cv;
#include "framediff.hpp"
#include "debug.hpp"

FrameDiff::FrameDiff( uint spsize ){
    spsz = spsize;
    queue.clear();
    return;
}

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
    cvtColor( queue[queue.size()-1], src2gray, COLOR_BGR2GRAY, 1 );
    subtract( src1gray, src2gray, dst, noArray(), 1 );
    return;
}