#include "SIFTFeatureDescripter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

const int SIFTFeatureDescripter::ORIENTATION_KERNEL_SIZE = 9;
const int SIFTFeatureDescripter::GLOBAL_ANGLE_RANGE = 10;

SIFTFeatureDescripter::SIFTFeatureDescripter(std::vector<std::pair<int, int>>& feature, cv::Mat& inputImg)
	:inputFeature(feature), img(cv::Mat(inputImg.rows, inputImg.cols, CV_8UC1))
{
	cv::cvtColor(inputImg, img, cv::COLOR_RGB2GRAY);
}

void SIFTFeatureDescripter::ComputeGlobalOrientation(int x, int y)
{
	cv::Mat subImage = Common::CropImg(img, x - ORIENTATION_KERNEL_SIZE / 2 - 1, y - ORIENTATION_KERNEL_SIZE / 2 - 1, ORIENTATION_KERNEL_SIZE + 2, ORIENTATION_KERNEL_SIZE + 2);
	const int BOX_LEN = 360 / GLOBAL_ANGLE_RANGE;
	float box[BOX_LEN] = {};
	for (int j = 1; j <= ORIENTATION_KERNEL_SIZE; j++) {
		for (int i = 1; i <= ORIENTATION_KERNEL_SIZE; i++) {
			float dx = subImage.at<char>(j, i + 1) - subImage.at<char>(j, i - 1);
			float dy = subImage.at<char>(j + 1, i) - subImage.at<char>(j - 1, i);
			float mag = std::sqrt(dx * dx + dy * dy);
			int ori = std::atan2f(dy, dx) * 180 / 3.14;
			if (ori < 0) ori += 360;
			int idx = ori / GLOBAL_ANGLE_RANGE;
			box[idx] += mag * (ori % GLOBAL_ANGLE_RANGE) / GLOBAL_ANGLE_RANGE;
			box[(idx + 1) % BOX_LEN] += mag * (GLOBAL_ANGLE_RANGE - ori % GLOBAL_ANGLE_RANGE) / GLOBAL_ANGLE_RANGE;
		}
	}
	float maxValue = 0;
	for (int i = 0; i < BOX_LEN; i++) {
		if (box[i] > maxValue) maxValue = box[i];
	}
	for (int i = 0; i < BOX_LEN; i++) {
		if (box[i] > maxValue * 0.8) {
			featureData.push_back(FeatureDescripterData({ x, y, i * GLOBAL_ANGLE_RANGE }));
		}
	}
}

double* SIFTFeatureDescripter::ComputeLocalDescriptor(int x, int y, int orientation)
{
	cv::Mat subImage = Common::CropImg(img, x - 13, y - 13, 27, 27);
	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f(13, 13), -orientation, 1);
	warpAffine(subImage, subImage, rotationMat, subImage.size());
	subImage = subImage(cv::Rect(4, 4, 18, 18));
	double *descriptorData = new double[128]{};
	for (int j = 1; j < 17; j++) {
		for (int i = 1; i < 17; i++) {
			int ii = i - 1, jj = j - 1;
			float dx = subImage.at<char>(j, i + 1) - subImage.at<char>(j, i - 1);
			float dy = subImage.at<char>(j + 1, i) - subImage.at<char>(j - 1, i);
			float mag = std::sqrt(dx * dx + dy * dy);
			int ori = std::atan2f(dy, dx) * 180 / 3.14;
			if (ori < 0) ori += 360;
			int idx = ((jj / 4) * 4 + (ii / 4)) * 8 + ori / 45;
			descriptorData[idx] += mag * Common::Gaussian(ii - 8, jj - 8, 8);
		}
	}
	// descriptorData[0] = orientation / 180;
	double total = 0;
	for (int i = 0; i < 128; i++) {
		total += descriptorData[i];
	}
	for (int i = 0; i < 128; i++) {
		descriptorData[i] /= total;
		descriptorData[i] = Common::Clip(descriptorData[i], 0.0, 0.2);
	}
	total = 0;
	for (int i = 0; i < 128; i++) {
		total += descriptorData[i];
	}
	for (int i = 0; i < 128; i++) {
		descriptorData[i] /= total;
	}
	return descriptorData;
}

std::vector<FeatureDescripterData> SIFTFeatureDescripter::Process(std::vector<std::pair<int, int>>& feature, cv::Mat &_img)
{
	cv::Mat img = _img.clone();
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	SIFTFeatureDescripter siftFeatureDescripter(feature, img);
	for (int i = 0; i < feature.size(); i++) {
		siftFeatureDescripter.ComputeGlobalOrientation(feature[i].first, feature[i].second);
	}
	std::vector<FeatureDescripterData> featureValues;
	for (int i = 0; i < siftFeatureDescripter.featureData.size(); i++) {
		double * featureValue = siftFeatureDescripter.ComputeLocalDescriptor(siftFeatureDescripter.featureData[i].x
			, siftFeatureDescripter.featureData[i].y
			, siftFeatureDescripter.featureData[i].orientation);
		siftFeatureDescripter.featureData[i].value = featureValue;
	}
	return siftFeatureDescripter.featureData;
}


void SIFTFeatureDescripter::Match(std::vector<FeatureDescripterData>& featureValues1, std::vector<FeatureDescripterData>& featureValues2) {
	FeatureDescripterDatas data;
	data.pts = featureValues2;
	KDTree index1(128, data);
	index1.buildIndex();

	std::vector<size_t>  retIndex(2);
	std::vector<double> outDistSqr(2);
	for (int i = 0; i < featureValues1.size(); i++) {
		bool found = index1.knnSearch(&featureValues1[i].value[0], 2, &retIndex[0], &outDistSqr[0]) >= 2;
		if (found) {
			int diffY = std::abs(featureValues1[i].y - featureValues2[retIndex[0]].y);
			if (diffY > 25) continue;
			double dis1 = Common::Distance<double>(featureValues1[i].value, featureValues2[retIndex[0]].value, 128);
			double dis2 = Common::Distance<double>(featureValues1[i].value, featureValues2[retIndex[1]].value, 128);
			if (dis2 >= 0.3 * dis1) continue;
			featureValues1[i].matchPoint = &featureValues2[retIndex[0]];
		}
	}
}
