#include "HarrisCornerDetection.h"
#include "Common.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

PointData::PointData(int x, int y, float weight): x(x), y(y), weight(weight)
{
}

bool PointData::operator<(const PointData& p) const
{
	if (weight != p.weight) return weight >= p.weight;
	else if (x != p.x) return x >= p.x;
	return y >= p.y;
}

std::vector<std::pair<int, int>>  HarrisCornerDetection::Process(cv::Mat &img)
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
	std::vector<PointData> points;
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			float det = Det.at<float>(y, x);
			float tr = Tr.at<float>(y, x);
			/*float Tr = a + b;
			float Det = a * b - c * c;
			float R = Det - k * Tr * Tr;*/
			float r = R.at<float>(y, x);
			points.push_back(PointData(x, y, r));
			/*if (r > 8000000000) {
				cv::circle(corner, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
			}
			else if (r < -2000000){
				cv::circle(edge, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
			}*/
			/*if (det/tr > 100) {
				cv::circle(corner, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
			}*/
		}
	}
	std::sort(points.begin(), points.end());
	int r = img.rows + img.cols;
	const int featureTotal = 80;
	int featurePointIdxs[featureTotal];
	std::vector<std::pair<int, int>> featurePoints;
	for (int i = 0; i < featureTotal; i++) {
		int newFeaturePointIdx = -1;
		while (newFeaturePointIdx < 0) {
			for (int j = 0; newFeaturePointIdx < 0 && j < points.size() && (j < points.size() * 0.03 || j < newFeaturePointIdx * 10); j++) {
				bool valid = true;
				const PointData& p = points[j];
				for (int k = 0; k < i; k++) {
					const PointData& fp = points[featurePointIdxs[k]];
					float dis_2 = (p.x - fp.x) * (p.x - fp.x) + (p.y - fp.y) * (p.y - fp.y);
					if (dis_2 < r * r) {
						valid = false;
						break;
					}
				}
				if (valid) {
					newFeaturePointIdx = j;
					cv::circle(corner, cv::Point(p.x, p.y), 1, cv::Scalar(255, 0, 0), -1);
					featurePoints.push_back(std::pair<int, int>(p.x, p.y));
				}
			}
			if (newFeaturePointIdx < 0) r *= 0.97;
		}
		featurePointIdxs[i] = newFeaturePointIdx;
	}
	std::cout << "r: " << r << std::endl;
	cv::imwrite("gradientX.png", Ix);
	cv::imwrite("gradientY.png", Iy);
	cv::imwrite("corner.png", corner);
	return featurePoints;
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