#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <utility>

const int WEIGHT_KERNEL_SIZE = 5; //©_¼Æ

class PointData {
public:
	int x;
	int y;
	float weight;
	PointData(int x=0, int y=0, float weight=0);
	bool operator< (const PointData& p) const;
};

class HarrisCornerDetection
{
	static float GuassianFunc(float u, float v, float sigma = 1);
	static cv::Mat CreateWeightImg(int size = WEIGHT_KERNEL_SIZE);
public:
	static std::vector<std::pair<int, int>> Process(cv::Mat &img);

};

