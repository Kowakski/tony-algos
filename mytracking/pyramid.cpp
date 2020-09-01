/*
 *@使用opencv的接口显示效果，然后实现该函数对比
 */
/*
 *@pydonw 的过程:
 *1) 这个过程是先用高斯模糊图像
 *2) 去掉偶数的行和列
 */
/*
 Robert[2][2]={
 {+1,0},
 {0,-1}}

 Robert[2][2]={
 {0,+1},
 {-1,0}
 }


 Sobel[3][3]={
 {-1,0,1},
 {-2,0,2},
 {-1,0,1}
 }

 ={
 {-1,-2,-1},
 {0,  0, 0},
 {1,  2, 1}
 }

 Laplance[3][3]={
 {0,-1, 0},
 {-1,4,-1},
 {0,-1, 0}
 }
 */
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// #include "core/core.hpp"  
// #include "highgui/highgui.hpp"  
// #include "imgproc/imgproc.hpp"  
// #include "iostream"
 
using namespace std; 
using namespace cv;  

void print_img_vec2s( const Mat& img ){
    int i = 0;
    for (auto it = img.begin<short>(); it != img.end<short>(); ++it){
        i++;
        std::cout << int((*it)) << " ";
        if( i%img.cols == 0 ) cout << endl;
    }
}

#if 0

int main(int argc,char *argv[])  
{
 Mat image=imread(argv[1],0);
 Mat imageX=Mat::zeros(image.size(),CV_16SC1);
 Mat imageY=Mat::zeros(image.size(),CV_16SC1); 
 Mat imageXY=Mat::zeros(image.size(),CV_16SC1); 
 Mat imageX8UC;
 Mat imageY8UC;
 Mat imageXY8UC;
 if(!image.data)
 {
  return -1;
 }
 GaussianBlur(image,image,Size(3,3),0); 
 uchar *P=image.data;
 uchar *PX=imageX.data;
 uchar *PY=imageY.data;
 int step=image.step;
 int stepXY=imageX.step;
 for(int i=1;i<image.rows-1;i++)
 {
  for(int j=1;j<image.cols-1;j++)
  {
   //通过指针遍历图像上每一个像素
   PX[i*imageX.step+j*(stepXY/step)]=abs(P[(i-1)*step+j+1]+P[i*step+j+1]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[i*step+j-1]*2-P[(i+1)*step+j-1]);
   PY[i*imageX.step+j*(stepXY/step)]=abs(P[(i+1)*step+j-1]+P[(i+1)*step+j]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1]);
  }
 }
 addWeighted(imageX,0.5,imageY,0.5,0,imageXY);//融合X、Y方向 
 convertScaleAbs(imageX,imageX8UC);
 convertScaleAbs(imageY,imageY8UC);
 convertScaleAbs(imageXY,imageXY8UC);   //转换为8bit图像
 
 Mat imageSobel;
 Sobel(image,imageSobel,CV_8UC1,1,1); //Opencv的Sobel函数
 
 imshow("Source Image",image);
 imshow("X Direction",imageX8UC);
 imshow("Y Direction",imageY8UC);
 imshow("XY Direction",imageXY8UC);
 imshow("Opencv Soble",imageSobel);
 while( waitKey(10) != 'q' ){}
 return 0;
}

#else
void mySoble( const Mat& src, Mat& dst ){
  if( src.type() != CV_8UC1 ){
      cout << "src type error!"<<endl;
      return;
  }
  Mat srcImg = src.clone();
  Mat imgX  = Mat::zeros( src.size(), CV_16SC1 );
  Mat imgY  = Mat::zeros( src.size(), CV_16SC1 );
  Mat imgXY  = Mat::zeros( src.size(), CV_16SC1 );
  Mat imgUx;
  Mat imgUy;
  Mat imgUxy;
  GaussianBlur( srcImg, srcImg, Size(3,3),0 );
  uchar* srcP = srcImg.data;
  uchar* X = imgX.data;
  uchar* Y = imgY.data;
  for( int i = 1; i < srcImg.rows-1; i++ ){
   for( int j = 1; j < srcImg.cols-1; j++ ){
    X[ i*imgX.step[0] + j*imgX.step[1] ] = abs(srcP[(i-1)*srcImg.step+(j-1)*(srcImg.step/srcImg.cols)]*(-1) + srcP[(i-1)*srcImg.step+(j+1)*(srcImg.step/srcImg.cols)]*(1) + srcP[(i)*srcImg.step+(j-1)*(srcImg.step/srcImg.cols)]*(-2) + srcP[(i)*srcImg.step+(j+1)*(srcImg.step/srcImg.cols)]*(2)
                                                          + srcP[(i+1)*srcImg.step+(j-1)*(srcImg.step/srcImg.cols)]*(-1) + srcP[(i+1)*srcImg.step+(j+1)*(srcImg.step/srcImg.cols)]*(1) );

    Y[ i*imgY.step[0] + j*imgY.step[1] ] = abs(srcP[(i-1)*srcImg.step+(j-1)*(srcImg.step/srcImg.cols)]*(-1) + srcP[(i-1)*srcImg.step+(j)*(srcImg.step/srcImg.cols)]*(-2) + srcP[(i-1)*srcImg.step+(j+1)*(srcImg.step/srcImg.cols)]*(-1) + srcP[(i+1)*srcImg.step+(j-1)*(srcImg.step/srcImg.cols)]*(1)
                                                         + srcP[(i+1)*srcImg.step+(j)*(srcImg.step/srcImg.cols)]*(2) + srcP[(i+1)*srcImg.step+(j+1)*(srcImg.step/srcImg.cols)]*(1) );
   }

  }
     // imshow("imgX", imgX);
     // convertScaleAbs(imgX,imgUx);
     // imshow("imgUx", imgUx);
     // convertScaleAbs(imgY,imgUy);
    addWeighted(imgX,0.5,imgX,0.5,0,imgXY);
    convertScaleAbs( imgXY, imgUxy );
    dst = imgUxy.clone();
    return;
}

template<typename T>
void prymid_( const Mat& src, Mat& dst ){
    T* srcptr          = NULL;
    T* dstptr          = NULL;

    int cn = src.channels();
    dst.create( src.rows/2, src.cols/2, src.type() );
    Mat temp = src.clone();
    GaussianBlur(temp,temp,Size(3,3),0);

    dstptr = (T*)dst.data;

    for(  int i = 0; i < src.rows; i += 2 ){
      for( int j = 0; j < src.cols; j += 2 ){
        srcptr = (T*)src.data + i*src.step[0] + j*src.step[1];
        switch( cn ){
          case 1:
            dstptr[0] = srcptr[0];
          break;

          case 2:
            dstptr[0] = srcptr[0];

            dstptr[1] = srcptr[1];
          break;

          case 3:
            dstptr[0] = srcptr[0];

            dstptr[1] = srcptr[1];

            dstptr[2] = srcptr[2];
          break;
        }
      }
    }

    return;

}

void myprymid( const Mat& src, Mat& dst ){
 void (*func)(  const Mat& src, Mat& dst  ) = NULL;
 switch( src.depth() ){
  case CV_8U:
      func = prymid_<uchar>;
  break;

  case CV_16U:
     func = prymid_<ushort>;
  break;
 }
 func( src, dst );
}

int main( int argc, char*argv[] ){
    Mat srcImage = imread( "./001.png", IMREAD_COLOR  ); //read as BGR
    if( srcImage.empty() ){
        cout << "read img error!!"<<endl;
        exit(1);
    }
    Mat oppyr1, oppyr2, oppyr3;

    Mat mm( 4, 4, CV_8UC3, Scalar_<uchar>(1,2,3) );
    Mat mmdst;
    myprymid( mm, mmdst );
    pyrDown( srcImage, oppyr1 );
    // imshow("pyr1",oppyr1);
    Mat prop( 4,5,CV_16UC3, Scalar(1,2,3) );
    pyrDown( oppyr1, oppyr2 );
    imshow("pyr2", oppyr2);
    Mat src_gray, grad_x, grad_y, dst;
    cvtColor( oppyr2, src_gray, COLOR_BGR2GRAY );
    mySoble( src_gray, dst );
    print_img_vec2s(dst);
    imshow("mysobel", dst);
    // Sobel( src_gray, grad_x,  )
    // pyrDown( oppyr2, oppyr3 );
    // imshow("pyr3", oppyr3);
    while( waitKey(100) != 'q' ){}
    return 0;
}

#endif