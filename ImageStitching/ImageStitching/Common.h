#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

const int fSize = 11;

struct FeatureDescriptor {
	int x;
	int y;
	double* value;					// (fSize / 2)^2 dims
	double diff;
	FeatureDescriptor* matchPoint;
};

class Common
{
public:
	static cv::Mat GetGradientX(cv::Mat& img);
	static cv::Mat GetGradientY(cv::Mat& img);

	static cv::Mat ProductEveryPixel(cv::Mat& img1, cv::Mat& img2);
	static cv::Mat Convolution(cv::Mat& img, cv::Mat& kernel);
	static cv::Mat CropImg(cv::Mat& img, int x, int y, int width, int height);
	static void imshow(cv::Mat& img);
	template<typename T>
	static T Clip(T n, T lower, T upper);
	template<typename T>
	static bool InRange(T n, T lower, T upper);
	template<typename T>
	static double Distance(T* arr1, T* arr2, int dim);
	static double Gaussian(double x, double y, double sigma = 1);

	// Find Feature Descriptor & Match Pair
	static std::vector<FeatureDescriptor> Process(std::vector<std::pair<int, int>>& feature, cv::Mat& img);
	static void Match(std::vector<FeatureDescriptor>& featureValues1, std::vector<FeatureDescriptor>& featureValues2);
	static void MatchFilter(std::vector<FeatureDescriptor>& featureValues);
};

template<typename T>
T Common::Clip(T n, T lower, T upper)
{
	if (n < lower) return lower;
	if (n > upper) return upper;
	return n;
}

template<typename T>
bool Common::InRange(T n, T lower, T upper)
{
	if (n < lower || n > upper) return false;
	return true;
}

template<typename T>
double Common::Distance(T* arr1, T* arr2, int dim)
{
	double total = 0;
	for (int i = 0; i < dim; i++) {
		double diff = arr1[0] - arr2[1];
		total += diff * diff;
	}
	return std::sqrt(total);
}

