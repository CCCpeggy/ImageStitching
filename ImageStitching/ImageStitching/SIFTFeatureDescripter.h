#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <utility>
#include <nanoflann.hpp>
#include "Common.h"

struct FeatureDescripterData {
	int x;
	int y;
	int orientation;
	double* value; // 128 dim
	FeatureDescripterData* matchPoint;
};

struct FeatureDescripterDatas
{
	std::vector<FeatureDescripterData>  pts;

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline double kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return pts[idx].value[dim];
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<double, FeatureDescripterDatas>,
	FeatureDescripterDatas, 128> KDTree;

class SIFTFeatureDescripter
{
private:
	static const int ORIENTATION_KERNEL_SIZE;
	static const int GLOBAL_ANGLE_RANGE;
	static const int DIM;
	std::vector<std::pair<int, int>>& inputFeature;
	std::vector<FeatureDescripterData> featureData;
	cv::Mat img;
	SIFTFeatureDescripter(std::vector<std::pair<int, int>>& feature, cv::Mat& img);
	void ComputeGlobalOrientation(int x, int y);
	double* ComputeLocalDescriptor(int x, int y, int orientation);

public:
	static std::vector<FeatureDescripterData> Process(std::vector<std::pair<int, int>>& feature, cv::Mat& img);
	static void Match(std::vector<FeatureDescripterData>& featureValues1, std::vector<FeatureDescripterData>& featureValues2);
};

