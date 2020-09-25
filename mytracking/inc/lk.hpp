#include <opencv2/opencv.hpp>

class lk{
    public:
         lk( void );
         ~lk( void );
         inline void clear_init_tarck_points( void ){
              initTrackPoints.clear();
              return;
         }
    private:
         std::vector<Point2f> initTrackPoints;
};