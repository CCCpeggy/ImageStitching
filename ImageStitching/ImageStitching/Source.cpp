#include "HarrisCornerDetection.h"
#include "Common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
	cv::Mat img = cv::imread("../../Images/test/edge_dectection_guassian.png");
	assert(!img.empty());
	

	std::vector<std::pair<int, int>> featurePoints = HarrisCornerDetection::Process(img);

}