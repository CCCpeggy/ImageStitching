#pragma once

#include <opencv2/core.hpp>

const int WEIGHT_KERNEL_SIZE = 5; //©_¼Æ
class HarrisCornerDetection
{
	static float GuassianFunc(float u, float v, float sigma = 1);
	static cv::Mat CreateWeightImg(int size = WEIGHT_KERNEL_SIZE);
public:
	static void Process(cv::Mat &img);

};

