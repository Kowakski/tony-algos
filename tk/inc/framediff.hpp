class FrameDiff{
public:
    FrameDiff( Mat inputarray );
private:
    uint spsz;  //span size
    std::vector<Mat> queue;
}