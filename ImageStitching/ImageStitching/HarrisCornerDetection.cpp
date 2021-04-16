#include "HarrisCornerDetection.h"
#include "Common.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>

static int imgNo = 0;

PointData::PointData(int x, int y, float weight): x(x), y(y), weight(weight)
{
}

bool PointData::operator<(const PointData& p) const
{
	if (weight != p.weight) return weight >= p.weight;
	else if (x != p.x) return x >= p.x;
	return y >= p.y;
}

std::vector<std::pair<int, int>>  HarrisCornerDetection::Process(cv::Mat& _img)
{
	cv::Mat img = _img.clone();
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	cv::Mat corner = _img.clone();
	cv::Mat feature = _img.clone();
	cv::Mat grayImg = cv::Mat(img.rows, img.cols, CV_8UC1);
	cv::cvtColor(img, grayImg, cv::COLOR_RGB2GRAY);
	cv::Mat w = CreateWeightImg();

	/*cv::Mat rgbImg[3];
	cv::split(img, rgbImg);
	cv::Mat rgbImgX[3] = { Common::GetGradientX(rgbImg[0]),Common::GetGradientX(rgbImg[1]), Common::GetGradientX(rgbImg[2]) };
	cv::Mat rgbImgY[3] = { Common::GetGradientY(rgbImg[0]),Common::GetGradientY(rgbImg[1]), Common::GetGradientY(rgbImg[2]) };*/
	cv::Mat Ix = Common::GetGradientX(grayImg);
	cv::Mat Iy = Common::GetGradientY(grayImg);
	// for (int j = 0; j < img.rows; j++) {
	// 	for (int i = 0; i < img.cols; i++) {
	// 		Ix.at<float>(j, i) = std::max(std::max(rgbImgX[0].at<float>(j, i), rgbImgX[1].at<float>(j, i)), rgbImgX[2].at<float>(j, i));
	// 		Iy.at<float>(j, i) = std::max(std::max(rgbImgY[0].at<float>(j, i), rgbImgY[1].at<float>(j, i)), rgbImgY[2].at<float>(j, i));
	// 	}
	// }
	//Ix = Ix * 0.5 + Common::GetGradientX(grayImg) * 0.8;
	// cv::Mat Iy = (Common::GetGradientY(rgbImg[0]) + Common::GetGradientY(rgbImg[1]) + Common::GetGradientY(rgbImg[2])) / 3;
	// cv::Mat Iy = Common::GetGradientY(img);
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
	const int maxSize = 9;
	const int margin = 3;
	const int featureTotal =  300;
	const int threashold = 100000000;
	for (int x = margin; x < img.cols - margin; x++) {
		for (int y = margin; y < img.rows - margin; y++) {
			//float det = Det.at<float>(y, x);
			//float tr = Tr.at<float>(y, x);
			/*float Tr = a + b;
			float Det = a * b - c * c;
			float R = Det - k * Tr * Tr;*/
			float r = R.at<float>(y, x);
			if (r > threashold) {
				bool isLocalMax = true;
				for (int j = 0; isLocalMax && j < maxSize; j++) {
					for (int i = 0; isLocalMax && i < maxSize; i++) {
						int newX = x + i - maxSize / 2, newY = j + y - maxSize / 2;
						if (!Common::InRange(newX, 0, img.cols - 1)) continue;
						if (!Common::InRange(newY, 0, img.rows - 1)) continue;
						float rr = R.at<float>(newY, newX);
						if (rr > r) {
							isLocalMax = false;
						}
					}
				}
				if (isLocalMax) {
					points.push_back(PointData(x, y, r));
					cv::circle(corner, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
				}
			}
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
	std::cout << "found point: " << points.size() << std::endl;
	std::sort(points.begin(), points.end());
	int r = std::max(img.rows, img.cols) * 0.5;
	int featurePointIdxs[featureTotal];
	std::vector<std::pair<int, int>> featurePoints;
	for (int i = 0; i < featureTotal; i++) {
		int newFeaturePointIdx = -1;
		while (newFeaturePointIdx < 0) {
			for (int j = 0; newFeaturePointIdx < 0 && j < points.size() /*&& (j < points.size() * 0.5 || j < newFeaturePointIdx * 40)*/; j++) {
				const PointData& p = points[j];
				bool valid = true;
				for (int k = 0; k < i; k++) {
					const PointData& fp = points[featurePointIdxs[k]];
					float dis_2 = (p.x - fp.x) * (p.x - fp.x) + (p.y - fp.y) * (p.y - fp.y);
					if (dis_2 < r * r ) {
						valid = false;
						break;
					}
				}
				if (valid) {
					newFeaturePointIdx = j;
					cv::circle(feature, cv::Point(p.x, p.y), 1, cv::Scalar(255, 0, 0), -1);
					featurePoints.push_back(std::pair<int, int>(p.x, p.y));
				}
			}
			if (newFeaturePointIdx < 0) r *= 0.7;
		}
		featurePointIdxs[i] = newFeaturePointIdx;
	}
	std::cout << "r: " << r << std::endl;
	cv::imwrite(std::to_string(imgNo) + "_gradientX.png", Ix);
	cv::imwrite(std::to_string(imgNo) + "_gradientY"".png", Iy);
	cv::imwrite(std::to_string(imgNo) + "_corner.png", corner);
	cv::imwrite(std::to_string(imgNo) + "_feature.png", feature);
	imgNo += 1;
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