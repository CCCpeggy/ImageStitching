#include "HarrisCornerDetection.h"
#include "SIFTFeatureDescripter.h"
#include "Common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#define HOME_TEST 

int main() {
#ifdef DENNY_TEST
	cv::Mat img[] = {
		cv::imread("../../Images/test/denny/denny00.jpg"),
		cv::imread("../../Images/test/denny/denny01.jpg"),
		cv::imread("../../Images/test/denny/denny02.jpg"),
		cv::imread("../../Images/test/denny/denny03.jpg"),
		cv::imread("../../Images/test/denny/denny04.jpg"),
		cv::imread("../../Images/test/denny/denny05.jpg"),
		cv::imread("../../Images/test/denny/denny06.jpg"),
		cv::imread("../../Images/test/denny/denny07.jpg"),
		cv::imread("../../Images/test/denny/denny08.jpg"),
		cv::imread("../../Images/test/denny/denny09.jpg"),
		cv::imread("../../Images/test/denny/denny10.jpg"),
		cv::imread("../../Images/test/denny/denny11.jpg"),
		cv::imread("../../Images/test/denny/denny12.jpg"),
		cv::imread("../../Images/test/denny/denny13.jpg"),
		cv::imread("../../Images/test/denny/denny14.jpg"),
	};
	const int size = 2;
#endif // DENNY_TEST
#ifdef GRAIL_TEST
	cv::Mat img[] = {
		cv::imread("../../Images/test/grail/grail00.jpg"),
		cv::imread("../../Images/test/grail/grail01.jpg"),
		cv::imread("../../Images/test/grail/grail02.jpg"),
		cv::imread("../../Images/test/grail/grail03.jpg"),
		cv::imread("../../Images/test/grail/grail04.jpg"),
		cv::imread("../../Images/test/grail/grail05.jpg"),
		cv::imread("../../Images/test/grail/grail06.jpg"),
		cv::imread("../../Images/test/grail/grail07.jpg"),
		cv::imread("../../Images/test/grail/grail08.jpg"),
		cv::imread("../../Images/test/grail/grail09.jpg"),
		cv::imread("../../Images/test/grail/grail10.jpg"),
		cv::imread("../../Images/test/grail/grail11.jpg"),
		cv::imread("../../Images/test/grail/grail12.jpg"),
		cv::imread("../../Images/test/grail/grail13.jpg"),
		cv::imread("../../Images/test/grail/grail14.jpg"),
		cv::imread("../../Images/test/grail/grail15.jpg"),
		cv::imread("../../Images/test/grail/grail16.jpg"),
		cv::imread("../../Images/test/grail/grail17.jpg"),
	};
	const int size = 3;
#endif // GRAIL_TEST
#ifdef CSIE_TEST
	cv::Mat img[] = {
		cv::imread("../../Images/test/csie/csie00.jpg"),
		cv::imread("../../Images/test/csie/csie01.jpg"),
		cv::imread("../../Images/test/csie/csie02.jpg"),
		cv::imread("../../Images/test/csie/csie03.jpg"),
		cv::imread("../../Images/test/csie/csie04.jpg"),
		cv::imread("../../Images/test/csie/csie05.jpg"),
		cv::imread("../../Images/test/csie/csie06.jpg"),
		cv::imread("../../Images/test/csie/csie07.jpg"),
		cv::imread("../../Images/test/csie/csie08.jpg"),
	};
	const int size = 2;
#endif // GRAIL_TEST
#ifdef HOME_TEST
	cv::Mat img[] = {
		//cv::imread("../../Images/test/home/home01.jpg"),
		//cv::imread("../../Images/test/home/home02.jpg"),
		//cv::imread("../../Images/test/home/home03.jpg"),
		//cv::imread("../../Images/test/home/home04.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing01.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing02.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing03.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing04.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing05.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing06.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing07.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing08.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing09.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing10.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing11.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing12.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_1/Mt.Qixing13.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing01.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing02.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing03.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing04.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing05.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing06.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing07.jpg"),
		//cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing08.jpg"),
	};
	const int size = 4;
#endif // HOME_TEST

	// 最後結果圖
	cv::Mat result;
	// 水平連接多張照片
	cv::hconcat(img, size, result);

	// 每張圖片的特徵點
	//std::vector<std::vector<FeatureDescripterData>> featureValues;
	std::vector<std::vector<FeatureDescriptor>> featureValues;
	// 跌代每張圖片
	for (int i = 0; i < size; i++) {
		// 透過 Harris Corner Alogrithm 去找出每張圖片的特徵點
		// 用pair<int, int>(x, y)記下其位置，並存入[featurePoints]中
		// 另外圖片的尺寸大小可能會影響到特徵點的搜尋
		std::vector<std::pair<int, int>> featurePoints = HarrisCornerDetection::Process(img[i]);
		// 將[featurePoints]與對應的照片傳給SIFT Alogrithm
		// 讓它去描述每個特徵點
		//featureValues.push_back(SIFTFeatureDescripter::Process(featurePoints, img[i]));
		featureValues.push_back(Common::Process(featurePoints, img[i]));
	}

	// blending的程式碼，還沒整理
	//cv::Mat result2(img[0].rows, img[0].cols * size, img[0].type());
	//for (int y = 0; y < img[0].rows; y++) {
	//	for (int x = 0; x < img[0].cols; x++) {
	//		for (int c = 0; c < 3; c++)
	//			result2.at<cv::Vec3b>(y, x)[c] = img[0].at<cv::Vec3b>(y, x)[c];
	//	}
	//}
	//int dx = 0, dy = 0;
	//for (int i = 1; i < size; i++) {
	//	Common::Match(featureValues[i - 1], featureValues[i]);
	//	Common::MatchFilter(featureValues[i - 1]);
	//	std::vector<FeatureDescriptor> mData;
	//	for (int j = 0; j < featureValues[i - 1].size(); j++) {
	//		if (featureValues[i - 1][j].matchPoint != nullptr)
	//			mData.push_back(featureValues[i - 1][j]);
	//	}
	//	FeatureDescriptor temp = mData[rand() % mData.size()];
	//	int localDx = temp.x - temp.matchPoint->x;
	//	int localDy = temp.y - temp.matchPoint->y;
	//
	//	for (int y = 0; y < img[i].rows; y++) {
	//		for (int x = 0; x < img[i].cols; x++) {
	//			int newX = x + dx + localDx;
	//			int newY = y + dy + localDy;
	//			if (newX < result2.cols && newX >= 0 &&
	//				newY < result2.rows && newY >= 0) {
	//
	//				cv::Scalar mColor = result2.at<cv::Vec3b>(newY, newX);
	//				if (mColor != cv::Scalar(0, 0, 0)) {
	//					double alpha = (double)((img[i].cols + dx) - newX) / (double)(img[i].cols - localDx);
	//					if (alpha > 1)
	//						std::cout << alpha << std::endl;
	//					for (int c = 0; c < 3; c++) {
	//						result2.at<cv::Vec3b>(newY, newX)[c] =
	//							result2.at<cv::Vec3b>(newY, newX)[c] * alpha +
	//							img[i].at<cv::Vec3b>(y, x)[c] * (1 - alpha);
	//					}
	//				}
	//				else
	//					for (int c = 0; c < 3; c++)
	//						result2.at<cv::Vec3b>(newY, newX)[c] = img[i].at<cv::Vec3b>(y, x)[c];
	//			}
	//		}
	//	}
	//	dx += localDx;
	//	dy += localDy;
	//}
	//// 輸出result.png
	//cv::imwrite("result.png", result2);
	//return 0;

	// 預設照片的順序是從左到右排好
	// 在已經有特徵點的情況下
	// 去找img[i]與img[i+1]的對應特徵點
	// 透過k-d tree加速搜尋
	int shift1 = 0;
	int shift2 = 0;
	for (int i = 0; i < size - 1; i++) {
		shift2 += img[i].cols;
		// 傳入兩張照片的特徵描述子，找Match Pair
		//SIFTFeatureDescripter::Match(featureValues[i], featureValues[i + 1]);
		Common::Match(featureValues[i], featureValues[i + 1]);
		// 過濾img[i]的Match Pair
		//SIFTFeatureDescripter::MatchFilter(featureValues[i]);
		Common::MatchFilter(featureValues[i]);

		// 用line把Match Pair連起來
		for (int j = 0; j < featureValues[i].size(); j++) {
			if (featureValues[i][j].matchPoint) {
				cv::line(result
					, cv::Point2i(featureValues[i][j].x + shift1, featureValues[i][j].y)
					, cv::Point2i(featureValues[i][j].matchPoint->x + shift2, featureValues[i][j].matchPoint->y)
					//, cv::Scalar(std::rand() % 256, std::rand() % 128, std::rand() % 256), 1);
					, cv::Scalar(255, 31, 255), 1);
			}
		}
		shift1 += img[i].cols;
	}
	int shift = 0;
	// 把最後還留著的Match Point圈出來
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < featureValues[i].size(); j++) {
			cv::circle(result, cv::Point2i(featureValues[i][j].x + shift, featureValues[i][j].y)
				, 2, cv::Scalar(0, 255, 0));
		}
		shift += img[i].cols;
	}
	// 輸出result.png
	cv::imwrite("result.png", result);
	return 0;
}