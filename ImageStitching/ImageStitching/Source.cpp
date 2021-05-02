#include "HarrisCornerDetection.h"
#include "SIFTFeatureDescripter.h"
#include "Common.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
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
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing05.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing06.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing07.jpg"),
		cv::imread("../../Images/test/Mt.Qixing_2/Mt.Qixing08.jpg"),
	};
	const int size = 8;
#endif // HOME_TEST

	// cast to cylinder
	for (int i = 0; i < size; i++) {
		Common::ProjectToCylinder(img[i], img[i], 400);
	}

	// �̫ᵲ�G��
	cv::Mat result;
	// �����s���h�i�Ӥ�
	cv::hconcat(img, size, result);

	// �C�i�Ϥ����S�x�I
	//std::vector<std::vector<FeatureDescripterData>> featureValues;
	std::vector<std::vector<FeatureDescriptor>> featureValues;
	// �^�N�C�i�Ϥ�
	for (int i = 0; i < size; i++) {
		// �z�L Harris Corner Alogrithm �h��X�C�i�Ϥ����S�x�I
		// ��pair<int, int>(x, y)�O�U���m�A�æs�J[featurePoints]��
		// �t�~�Ϥ����ؤo�j�p�i��|�v�T��S�x�I���j�M
		std::vector<std::pair<int, int>> featurePoints = HarrisCornerDetection::Process(img[i]);
		// �N[featurePoints]�P�������Ӥ��ǵ�SIFT Alogrithm
		// �����h�y�z�C�ӯS�x�I
		//featureValues.push_back(SIFTFeatureDescripter::Process(featurePoints, img[i]));
		featureValues.push_back(Common::Process(featurePoints, img[i]));
	}

	for (int i = 0; i < size - 1; i++) {
		Common::Match(featureValues[i], featureValues[i + 1]);
		Common::MatchFilter(featureValues[i]);
	}

	//for (int i = 0; i < size - 1; i++) {
	//	cv::Mat h = Common::FindHomography(featureValues[i]);
	//	cv::Mat ttt = img[i].clone();
	//	cv::Mat out;
	//	cv::warpPerspective(img[i + 1], out, h, img[i].size());
	//
	//	for (int y = 0; y < img[i].rows; y++) {
	//		for (int x = 0; x < img[i].cols; x++) {
	//			cv::Scalar temp = out.at<cv::Vec3b>(y, x);
	//			cv::Scalar temp2 = ttt.at<cv::Vec3b>(y, x);
	//			if (temp == cv::Scalar(0, 0, 0)) {
	//				out.at<cv::Vec3b>(y, x) = ttt.at<cv::Vec3b>(y, x);
	//			}
	//			else if (temp2 != cv::Scalar(0, 0, 0) && temp != cv::Scalar(0, 0, 0)) {
	//				out.at<cv::Vec3b>(y, x) = out.at<cv::Vec3b>(y, x) * 0.5 +
	//					ttt.at<cv::Vec3b>(y, x) * 0.5;
	//			}
	//		}
	//	}
	//	cv::imwrite(std::to_string(i) + "_concat.png", out);
	//}
	//system("pause");
	//return 0;
	
	//for (int i = 0; i < size - 1; i++) {
	//	std::vector<cv::Point2f> pair1;
	//	std::vector<cv::Point2f> pair2;
	//	for (int j = 0; j < featureValues[i].size(); j++) {
	//		if (featureValues[i][j].matchPoint != nullptr) {
	//			pair1.push_back(cv::Point2f(featureValues[i][j].x, featureValues[i][j].y));
	//			pair2.push_back(cv::Point2f(
	//				featureValues[i][j].matchPoint->x,
	//				featureValues[i][j].matchPoint->y));
	//		}
	//	}
	//	cv::Mat ttt = img[i].clone();
	//	cv::Mat h = cv::findHomography(pair2, pair1);
	//	cv::Mat out;
	//	cv::warpPerspective(img[i + 1], out, h, img[i].size());
	//
	//	for (int y = 0; y < img[i].rows; y++) {
	//		for (int x = 0; x < img[i].cols; x++) {
	//			cv::Scalar temp = out.at<cv::Vec3b>(y, x);
	//			cv::Scalar temp2 = ttt.at<cv::Vec3b>(y, x);
	//			if (temp == cv::Scalar(0, 0, 0)) {
	//				out.at<cv::Vec3b>(y, x) = ttt.at<cv::Vec3b>(y, x);
	//			}
	//			else if (temp2 != cv::Scalar(0, 0, 0) && temp != cv::Scalar(0, 0, 0)) {
	//				out.at<cv::Vec3b>(y, x) = out.at<cv::Vec3b>(y, x) * 0.5 +
	//					ttt.at<cv::Vec3b>(y, x) * 0.5;
	//			}
	//		}
	//	}
	//	cv::imwrite(std::to_string(i) + "_concat.png", out);
	//}
	//return 0;

	// blending���{���X�A�٨S��z
	int width = img[0].cols;
	int height = img[0].rows;
	result = img[0].clone();
	int newWidth = width, newHeight = height;
	int dx = 0, dy = 0;
	for (int i = 1; i < size; i++) {
		cv::Mat h = Common::FindHomography(featureValues[i - 1]);
		int localDx = h.at<double>(0, 2);
		int localDy = h.at<double>(1, 2);
		newWidth = (newWidth + localDx) > newWidth ? (newWidth + localDx) : newWidth;
		newHeight = (newHeight + localDy) > newHeight ? (newHeight + localDy) : newHeight;
		// copy
		cv::Mat newImg(newHeight, newWidth, result.type());
		for (int r = 0; r < result.rows; r++) {
			for (int c = 0; c < result.cols; c++) {
				newImg.at<cv::Vec3b>(r, c) = result.at<cv::Vec3b>(r, c);
			}
		}
		newImg.copyTo(result);

		for (int y = 0; y < img[i].rows; y++) {
			for (int x = 0; x < width; x++) {
				int newX = x + dx + localDx;
				int newY = y + dy + localDy;
				if (newX < newWidth && newX >= 0 &&
					newY < newHeight && newY >= 0) {
	
					cv::Vec3b lResultC = result.at<cv::Vec3b>(newY, newX);
					cv::Vec3b lImgC = img[i].at<cv::Vec3b>(y, x);
					if (lResultC != cv::Vec3b(0, 0, 0) && lImgC != cv::Vec3b(0, 0, 0)) {
						double alpha = (((double)width + (double)dx) - (double)newX) / ((double)width - (double)localDx);
						alpha = pow(alpha, 2);
						result.at<cv::Vec3b>(newY, newX) =
							result.at<cv::Vec3b>(newY, newX) * alpha +
							img[i].at<cv::Vec3b>(y, x) * (1 - alpha);
					}
					else if (lImgC != cv::Vec3b(0, 0, 0))
						result.at<cv::Vec3b>(newY, newX) = img[i].at<cv::Vec3b>(y, x);
				}
			}
		}
		dx += localDx;
		dy += localDy;
	}
	// ��Xresult.png
	cv::imwrite("result.png", result);
	return 0;

	// �w�]�Ӥ������ǬO�q����k�Ʀn
	// �b�w�g���S�x�I�����p�U
	// �h��img[i]�Pimg[i+1]�������S�x�I
	// �z�Lk-d tree�[�t�j�M
	int shift1 = 0;
	int shift2 = 0;
	for (int i = 0; i < size - 1; i++) {
		shift2 += img[i].cols;
		// �ǤJ��i�Ӥ����S�x�y�z�l�A��Match Pair
		//SIFTFeatureDescripter::Match(featureValues[i], featureValues[i + 1]);
		Common::Match(featureValues[i], featureValues[i + 1]);
		// �L�oimg[i]��Match Pair
		//SIFTFeatureDescripter::MatchFilter(featureValues[i]);
		Common::MatchFilter(featureValues[i]);

		// ��line��Match Pair�s�_��
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
	// ��̫��ٯd�۪�Match Point��X��
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < featureValues[i].size(); j++) {
			cv::circle(result, cv::Point2i(featureValues[i][j].x + shift, featureValues[i][j].y)
				, 2, cv::Scalar(0, 255, 0));
		}
		shift += img[i].cols;
	}
	// ��Xresult.png
	cv::imwrite("result.png", result);
	return 0;
}