#include "SIFTFeatureDescripter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

const int SIFTFeatureDescripter::ORIENTATION_KERNEL_SIZE = 9;
const int SIFTFeatureDescripter::GLOBAL_ANGLE_RANGE = 10;
const int SIFTFeatureDescripter::DIM = 129;

SIFTFeatureDescripter::SIFTFeatureDescripter(std::vector<std::pair<int, int>>& feature, cv::Mat& inputImg)
	:inputFeature(feature), img(cv::Mat(inputImg.rows, inputImg.cols, CV_8UC1))
{
	cv::cvtColor(inputImg, img, cv::COLOR_RGB2GRAY);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
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
			/*box[idx] += mag * (GLOBAL_ANGLE_RANGE - ori % GLOBAL_ANGLE_RANGE) / GLOBAL_ANGLE_RANGE;
			box[(idx + 1) % BOX_LEN] += mag * (ori % GLOBAL_ANGLE_RANGE) / GLOBAL_ANGLE_RANGE;*/
			box[idx] += mag;
		}
	}
	int maxIdx = 0, secondMaxIdx = -1;
	for (int i = 1; i < BOX_LEN; i++) {
		if (box[i] > box[maxIdx]) {
			secondMaxIdx = maxIdx;
			maxIdx = i;
		}
		else if(secondMaxIdx <0 || box[i] > box[secondMaxIdx]){
			secondMaxIdx = i;
		}
	}
	featureData.push_back(FeatureDescripterData({ x, y, maxIdx * GLOBAL_ANGLE_RANGE }));
	if (box[secondMaxIdx] > box[maxIdx] * 0.8) {
		featureData.push_back(FeatureDescripterData({ x, y, secondMaxIdx * GLOBAL_ANGLE_RANGE }));
	}
}

double* SIFTFeatureDescripter::ComputeLocalDescriptor(int x, int y, int orientation)
{
	cv::Mat subImage = Common::CropImg(img, x - 13, y - 13, 27, 27);
	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f(13, 13), orientation, 1);
	warpAffine(subImage, subImage, rotationMat, subImage.size());
	subImage = subImage(cv::Rect(4, 4, 18, 18));
	double *descriptorData = new double[DIM]{};
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
	double total = 0;
	for (int i = 0; i < DIM; i++) {
		total += descriptorData[i];
	}
	float invDiv = 1 / total;
	for (int i = 0; i < DIM; i++) {
		descriptorData[i] *= invDiv;
		descriptorData[i] = Common::Clip(descriptorData[i], 0.0, 0.2);
	}
	total = 0;
	for (int i = 0; i < DIM; i++) {
		total += descriptorData[i] * descriptorData[i];
	}
	invDiv = 1 / std::sqrt(total);
	for (int i = 0; i < DIM; i++) {
		descriptorData[i] *= invDiv;
	}
	descriptorData[128] = std::sin(orientation * 3.14 / 180 / 2) * 0.01;
	return descriptorData;
}

std::vector<FeatureDescripterData> SIFTFeatureDescripter::Process(std::vector<std::pair<int, int>>& feature, cv::Mat &img)
{
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


void  SIFTFeatureDescripter::Match(std::vector<FeatureDescripterData>& featureValues1, std::vector<FeatureDescripterData>& featureValues2) {
	FeatureDescripterDatas data;
	data.pts = featureValues2;
	KDTree index1(DIM, data);
	index1.buildIndex();
	std::vector<std::pair<FeatureDescripterData*, FeatureDescripterData*> >  matchPoints;
	std::vector<size_t>  retIndex(2);
	std::vector<double> outDistSqr(2);
	for (int i = 0; i < featureValues1.size(); i++) {
		int foundLine = index1.knnSearch(&featureValues1[i].value[0], 2, &retIndex[0], &outDistSqr[0]);
		if (foundLine == 2) {
			//int diffY = std::abs(featureValues1[i].y - featureValues2[retIndex[0]].y);
			//if (diffY > 25) continue;
			//double dis1 = Common::Distance<double>(featureValues1[i].value, featureValues2[retIndex[0]].value, DIM);
			//double dis2 = Common::Distance<double>(featureValues1[i].value, featureValues2[retIndex[1]].value, DIM);
			//if (dis2 >= 0.8 * dis1) continue;
			if (outDistSqr[1] * 0.7 < outDistSqr[0]) continue;
			featureValues1[i].matchPoint = &featureValues2[retIndex[0]];
			matchPoints.push_back(std::pair<FeatureDescripterData*, FeatureDescripterData*>(&featureValues1[i], &featureValues2[retIndex[0]]));
		}
		else if (foundLine == 1){
			featureValues1[i].matchPoint = &featureValues2[retIndex[0]];
		}
	}
	std::vector<float> mag(matchPoints.size());
	std::vector<int> ori(matchPoints.size());
	std::vector<bool> exist(matchPoints.size(), true);
	float magTotal = 0, oriTotal = 0;
	for (int i = 0; i < matchPoints.size(); i++) {
		int dx = matchPoints[i].second->x - matchPoints[i].first->x;
		int dy = matchPoints[i].second->y - matchPoints[i].first->y;
		mag[i] = std::sqrt(dx * dx + dy * dy);
		ori[i] = std::atan2f(dy, dx) * 180 / 3.14;
		if (ori[i] < 0) ori[i] += 360;
		magTotal += mag[i];
		oriTotal += ori[i];
	}
	float magAvg = magTotal / matchPoints.size();
	float oriAvg = oriTotal / matchPoints.size();
	for (int i = 0; i < matchPoints.size(); i++) {
		if (mag[i] > magAvg * 1.8 || mag[i] < magAvg * 0.5) exist[i] = false;
		//if (ori[i] > oriAvg * 2 || ori[i] < oriAvg * 0.5) exist[i] = false;
	}
	
	int minI = -1;
	float minError = 0;
	for (int i = 0; i < matchPoints.size(); i++) {
		if (!exist[i]) continue;
		float error = 0;
		int dx = matchPoints[i].second->x - matchPoints[i].first->x;
		int dy = matchPoints[i].second->y - matchPoints[i].first->y;
		for (int j = 0; j < matchPoints.size(); j++) {
			if (!exist[j]) continue;
			if (i == j) continue;
			int dx2 = matchPoints[j].second->x - matchPoints[j].first->x;
			int dy2 = matchPoints[j].second->y - matchPoints[j].first->y;
			error += std::sqrt((dx - dx2) * (dx - dx2) + (dy - dy2) * (dy - dy2));
		}
		if (minI < 0 || error < minError) {
			minI = i;
			minError = error;
		}
	}
	std::cout << matchPoints[minI].first->x << " " << matchPoints[minI].first->y << std::endl;
	int dx = matchPoints[minI].second->x - matchPoints[minI].first->x;
	int dy = matchPoints[minI].second->y - matchPoints[minI].first->y;
	for (int j = 0; j < matchPoints.size(); j++) {
		if (!exist[j]) continue;
		if (minI == j) continue;
		int dx2 = matchPoints[j].second->x - matchPoints[j].first->x;
		int dy2 = matchPoints[j].second->y - matchPoints[j].first->y;
		float error = std::sqrt((dx - dx2) * (dx - dx2) + (dy - dy2) * (dy - dy2));
		if (error > 40) {
			exist[j] = false;
		}
	}
	for (int i = 0; i < matchPoints.size(); i++) {
		if (!exist[i]) {
			matchPoints[i].first->matchPoint = nullptr;
		}
	}
}
