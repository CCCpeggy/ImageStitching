#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

class Common
{
public:
	static cv::Mat GetGradientX(cv::Mat& img);
	static cv::Mat GetGradientY(cv::Mat& img);

	static cv::Mat ProductEveryPixel(cv::Mat& img1, cv::Mat& img2);
	static cv::Mat Convolution(cv::Mat& img, cv::Mat& kernel);
	static int Clip(int n, int lower, int upper);
};

