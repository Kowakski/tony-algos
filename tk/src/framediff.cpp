#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "framediff.hpp"
#include "debug.hpp"

FrameDiff::FrameDiff( uint spsize ){
    spsz = spsize;
    queue.clear();
    return;
}

void FrameDiff::FrameDiff( InputArray src1, OutputArray dst ){
    Mat src1gray, src2gray;
    cvtColor( src1, src1gray, COLOR_BGR2GRAY, 1 );

    if( queue.size() >= spsz ){
        queue.erase( queue.begin() );   //queue too long ,delete the eraliest one
    }

    queue.push_back( src1gray );

    if( queue.size() < spsz ){
        dst = noArray();
        dbg("FrameDiff queue is short\n");
        return;
    }
    cvtColor( queue[queue.size()-1], src2gray, COLOR_BGR2GRAY, 1 );
    subtract( src1gray, src2gray, dst, noArray(), 1 );
    return;
}