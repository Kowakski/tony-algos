class FrameDiff{
public:
    FrameDiff( uint spsize );
    void fd( InputArray src1, OutputArray dst );
private:
    uint spsz;  //span size
    std::vector<Mat> queue;
};