#include "HarrisCornerDetection.h"
#include "Common.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

void  HarrisCornerDetection::Process(cv::Mat &img)
{
	cv::Mat edge = img.clone();
	//cv::Mat edge = cv::Mat(img.rows, img.cols, CV_8UC3);
	cv::Mat corner = img.clone();
	//cv::Mat corner = cv::Mat(img.rows, img.cols, CV_8UC3);
	cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
	cv::Mat w = CreateWeightImg();

	cv::Mat Ix = Common::GetGradientX(img);
	cv::Mat Iy = Common::GetGradientY(img);
	cv::Mat Ix_2 = Ix.mul(Ix);
	cv::Mat Iy_2 = Iy.mul(Iy);
	cv::Mat IxIy = Ix.mul(Iy);

	const float k = 0.05;
	cv::Mat A = Common::Convolution(Ix_2, w);
	cv::Mat B = Common::Convolution(Iy_2, w);
	cv::Mat C = Common::Convolution(IxIy, w);
	cv::Mat Tr = A + B;
	cv::Mat Det = A.mul(B) - C.mul(C);
	cv::Mat R = Det - (Tr.mul(Tr)) * k;
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			float a = A.at<float>(y, x);
			float b = B.at<float>(y, x);
			float c = C.at<float>(y, x);
			float det = Det.at<float>(y, x);
			float tr = Tr.at<float>(y, x);
			/*float Tr = a + b;
			float Det = a * b - c * c;
			float R = Det - k * Tr * Tr;*/
			float r = R.at<float>(y, x);
			if (r > 8000000000) {
				cv::circle(corner, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
			}
			else if (r < -2000000){
				cv::circle(edge, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
			}
			if (det/tr > 100) {
				cv::circle(corner, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
			}
		}
	}
	cv::imwrite("gradientX.png", Ix);
	cv::imwrite("gradientY.png", Iy);
	cv::imwrite("corner.png", corner);
	cv::imwrite("edge.png", edge);
}



float HarrisCornerDetection::GuassianFunc(float u, float v, float sigma)
{
	return std::exp(-(u * u + v * v) / (2 * sigma * sigma));
}

cv::Mat HarrisCornerDetection::CreateWeightImg(int size)
{
	cv::Mat guassian = cv::Mat(size, size, CV_32FC1);
	int halfSize = size / 2;
	for (int x = 0; x <= halfSize; x++) {
		for (int y = 0; y <= halfSize; y++) {
			float gValue = GuassianFunc(x, y);
			guassian.at<float>(halfSize - y, halfSize - x) = gValue;
			guassian.at<float>(halfSize - y, halfSize + x) = gValue;
			guassian.at<float>(halfSize + y, halfSize - x) = gValue;
			guassian.at<float>(halfSize + y, halfSize + x) = gValue;
		}
	}
	//cv::imwrite("guassian.png", guassian);
	return guassian;
}
