/*
 *@comment:主要是验证cv::mat的一些结构和属性
 * */
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static void showImgPara(Mat &img)
{
	cout << "sizeof(img) is: " << sizeof(img) << ", img size is: " << img.size <<" img data size "<< sizeof(img.data) << endl;
	cout << "rows x cols: (" << img.rows << " x " << img.cols << ")" << endl;
	cout << "dims: " << img.dims << endl;
	cout << "channels: " << img.channels() << endl;
	cout << "type: " << img.type() << endl;
	cout << "depth:" << img.depth() << endl;
	cout << "elemSize:" << img.elemSize() << " (Bytes per element)" << endl;
	cout << "elemSize1:" << img.elemSize1() << "(Bytes per channel)" << endl;
	cout << "step[0]: " << img.step[0] << " (Bytes per cows only when 2 dims)" << endl;
	cout <<	"step[1]: " << img.step[1] << " (Bytes per element only when 2 dims)" << endl;
	cout << "step1(0): " << img.step1(0) << ", step1(1): " << img.step1(1) << " (step / elemSize1)" << endl;
}

static bool pictureTest(Mat &img, string name)
{
	if (img.empty())
	{
		cout << "Load picture fail!" << endl;
		return false;
	}
	cout << endl << "/******************pictureTest******************/" << endl;
	showImgPara(img);

	namedWindow(name);
	imshow(name, img);
	return true;
}

static bool matTest(Mat &img)
{
	uchar* pTemp = NULL;
	cout << endl << "/******************matTest******************/" << endl;
	cout << img << endl;
	showImgPara(img);

	for (int i = 0; i < img.rows; i++)
	{
		for(int j = 0; j < img.cols; j++)
		{
			cout << "[";
			for (int k = 0; k < img.step[1]; k++)
			{
				pTemp = img.data + img.step[0] * i + img.step[1] * j + k;
				cout << (int)*pTemp << " ";
			}
			cout << "] ";
		}
		cout << endl;
	}

	return true;
}

int main()
{
	Mat m2(3, 4, CV_8UC2, Scalar_<uchar>(1, 2));
#if 0
	Mat srcimg;
	srcimg = imread("/home/sln/share/draft/0001.jpg");
	if( srcimg.empty() ){
		cout << "Read fail !"<<endl;
	}
	matTest(srcimg);
#endif
	matTest(m2);

	waitKey(0);
	return 0;
}
