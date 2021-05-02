#include "Common.h"

cv::Mat Common::GetGradientX(cv::Mat& img)
{
	cv::Mat kernel = cv::Mat(1, 3, CV_32F, new float[1][3]{ {-1, 0, 1} });
	cv::Mat kernel2 = cv::Mat(3, 3, CV_32F, new float[3][3]{ {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} });
	cv::Mat gradient = Convolution(img, kernel);
	return gradient;

}

cv::Mat Common::GetGradientY(cv::Mat& img)
{
	cv::Mat kernel = cv::Mat(3, 1 , CV_32F, new float[3][1]{ {-1}, {0}, {1} });
	cv::Mat kernel2 = cv::Mat(3, 3, CV_32F, new float[3][3]{ {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} });
	cv::Mat gradient = Convolution(img, kernel);
	return gradient;
}

cv::Mat Common::ProductEveryPixel(cv::Mat& img1, cv::Mat& img2)
{
	assert(img1.rows == img2.rows && img1.cols == img2.cols && img1.type() == CV_32FC1 && img2.type() == CV_32FC1);
	cv::Mat product = cv::Mat(img1.rows, img1.cols, CV_32FC1);

	for (int i = 0; i < img1.cols; i++) {
		for (int j = 0; j < img1.rows; j++) {
			product.at<float>(j, i) = img1.at<float>(j, i) * img2.at<float>(j, i);
		}
	}

	return product;
}

cv::Mat Common::Convolution(cv::Mat& img, cv::Mat& kernel)
{
	assert((img.type() == CV_8UC1 || img.type() == CV_32FC1) && kernel.type() == CV_32FC1);
	cv::Mat result = cv::Mat(img.rows, img.cols, CV_32FC1);
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			float sum = 0;
			for (int i = 0; i < kernel.cols; i++) {
				for (int j = 0; j < kernel.rows; j++) {
					int newX = x + i - kernel.cols / 2, newY = j + y - kernel.rows / 2;
					newX = Clip(newX, 0, img.cols - 1);
					newY = Clip(newY, 0, img.rows - 1);
					if (img.type() == CV_8UC1)
						sum += img.at<char>(newY, newX) * kernel.at<float>(j, i);
					else
						sum += img.at<float>(newY, newX) * kernel.at<float>(j, i);
				}
			}
			result.at<float>(y, x) = std::abs(sum);
		}
	}
	return result;
}

cv::Mat Common::CropImg(cv::Mat& img, int x, int y, int width, int height)
{
	int left = x < 0 ? -x : 0;
	int top = y < 0 ? -y : 0;
	int right = (x + width) >= img.cols ? ((x + width) - img.cols) : 0;
	int down = (y + height) >= img.rows ? ((y + height) - img.rows) : 0;
	cv::Mat tmp = img(cv::Rect(x + left, y + top, width - left - right, height - top - down));
	cv::copyMakeBorder(tmp, tmp, top, down, left, right, cv::BORDER_REPLICATE);

	return tmp;
}

void Common::imshow(cv::Mat& img)
{
	cv::imshow("view", img);
	cv::waitKey(0);
}

double Common::Gaussian(double x, double y, double sigma)
{

	return 1.0 / (2.0 * 3.14 * sigma * sigma) * std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
}

std::vector<FeatureDescriptor> Common::Process(std::vector<std::pair<int, int>>& feature, cv::Mat& _img) {
	// ���N�����Ƕ�
	cv::Mat img = cv::Mat(_img.rows, _img.cols, CV_8UC1);
	cv::cvtColor(_img, img, cv::COLOR_RGB2GRAY);
	std::vector<FeatureDescriptor> mTempFeatDesc;
	for (int i = 0; i < feature.size(); i++) {
		// ���L��ɭ�
		if (feature[i].first - fSize / 2 < 0 || feature[i].first + fSize / 2 >= img.cols ||
			feature[i].second - fSize / 2 < 0 || feature[i].second + fSize / 2 >= img.rows)
			continue;
		// �p�G���O�����ɪ��S�x�I�A�~�����y�z�l
		FeatureDescriptor fd;
		double* tempValues = new double[fSize * fSize];
		for (int x = 0; x < fSize; x++) {
			for (int y = 0; y < fSize; y++) {
				int newX = feature[i].first + x - fSize / 2;
				int newY = feature[i].second + y - fSize / 2;
				newX = Clip<int>(newX, 0, img.cols - 1);
				newY = Clip<int>(newY, 0, img.rows - 1);

				tempValues[x * fSize + y] = img.at<uchar>(newY, newX);
			}
		}
		fd.x = feature[i].first;
		fd.y = feature[i].second;
		fd.value = tempValues;
		fd.matchPoint = nullptr;
		fd.diff = INFINITY;
		mTempFeatDesc.push_back(fd);
	}
	return mTempFeatDesc;
}

void Common::Match(std::vector<FeatureDescriptor>& fValues1, std::vector<FeatureDescriptor>& fValues2) {
	int kPixels = 5;
	for (int i = 0; i < fValues1.size(); i++) {
		double minOfDiff = INFINITY;
		int minIndex = -1;
		for (int j = 0; j < fValues2.size(); j++) {
			// ���]1: img[i]��x�@�w�|�j��img[i+1]��x, �]����ӮɬO�ѥ����k����
			if (fValues1[i].x <= fValues2[j].x)
				continue;
			// ���]2: �������q����W�L���t[kPixel]��px, �]���Ϥ����z�Q�W���|���Ӥj����������
			float dy = fValues1[i].y - fValues2[j].y;
			if (dy * dy > kPixels * kPixels)
				continue;
			// �p�G��Ӱ��]���ŦX, �h��X[1, fSize * fSize]���V�q���̤p�t�̬�pair
			double sumOfDiff = 0;
			for (int k = 0; k < fSize * fSize; k++) {
				double diff = fValues1[i].value[k] - fValues2[j].value[k];
				sumOfDiff += (diff * diff);
			}
			if (sumOfDiff < minOfDiff) {
				minOfDiff = sumOfDiff;
				minIndex = j;
			}
		}
		// ���pair
		if (minIndex != -1) {
			fValues1[i].matchPoint = &fValues2[minIndex];
			fValues1[i].diff = minOfDiff;
		}
	}
}

void Common::MatchFilter(std::vector<FeatureDescriptor>& featureValues) {
	/*
	int N = featureValues.size();
	double avg = 0;
	for (int i = 0; i < N; i++) {
		if (featureValues[i].matchPoint != nullptr)
			avg += featureValues[i].diff;
	}
	avg /= (double)N;
	for (int i = 0; i < N; i++) {
		if (featureValues[i].matchPoint != nullptr &&
			featureValues[i].diff > avg * 0.05) {
			featureValues[i].matchPoint = nullptr;
		}
	}
	*/
	
	std::vector<std::pair<FeatureDescriptor*, FeatureDescriptor*> >  matchPoints;
	for (int i = 0; i < featureValues.size(); i++) {
		if (featureValues[i].matchPoint) {
			matchPoints.push_back(std::pair<FeatureDescriptor*, FeatureDescriptor*>(&featureValues[i], featureValues[i].matchPoint));
		}
	}
	std::vector<float> mag(matchPoints.size());
	std::vector<int> ori(matchPoints.size());
	std::vector<int> oriBox(36);
	std::vector<bool> exist(matchPoints.size(), true);
	float magTotal = 0;
	for (int i = 0; i < matchPoints.size(); i++) {
		int dx = matchPoints[i].second->x - matchPoints[i].first->x;
		int dy = matchPoints[i].second->y - matchPoints[i].first->y;
		mag[i] = std::sqrt(dx * dx + dy * dy);
		ori[i] = std::atan2f(dy, dx) * 180 / 3.14;
		if (ori[i] < 0) ori[i] += 360;
		oriBox[((ori[i] + 5) / 10) % 36] ++;
	}
	// �L�o����
	int maxOriIdx = 0;
	for (int i = 1; i < 36; i++) {
		if (oriBox[i] > oriBox[maxOriIdx]) maxOriIdx = i;
	}
	for (int i = 0; i < matchPoints.size(); i++) {
		if (std::abs(ori[i] - maxOriIdx * 10) > 7) exist[i] = false;
	}
	// �L�o����
	int counter = 0;
	for (int i = 0; i < matchPoints.size(); i++) {
		if (exist[i]) {
			magTotal += mag[i];
			counter++;
		}
	}
	float magAvg = magTotal / counter;
	for (int i = 0; i < matchPoints.size(); i++) {
		if (mag[i] > magAvg * 2.2 || mag[i] < magAvg * 0.3) exist[i] = false;
	}

	// �N�Ҧ��i��ʶ]�@���A�ݭ��@�ػ~�t�ȳ̤p
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
			if (error > minError && minI >= 0) break;
		}
		if (minI < 0 || error < minError) {
			minI = i;
			minError = error;
		}
	}
	std::cout << matchPoints[minI].first->x << " " << matchPoints[minI].first->y << std::endl;

	// �N�~�t�Ȥj���L�o
	int dx = matchPoints[minI].second->x - matchPoints[minI].first->x;
	int dy = matchPoints[minI].second->y - matchPoints[minI].first->y;
	for (int j = 0; j < matchPoints.size(); j++) {
		if (!exist[j]) continue;
		if (minI == j) continue;
		int dx2 = matchPoints[j].second->x - matchPoints[j].first->x;
		int dy2 = matchPoints[j].second->y - matchPoints[j].first->y;
		float error = std::sqrt((dx - dx2) * (dx - dx2) + (dy - dy2) * (dy - dy2));
		if (error > 3) {
			exist[j] = false;
		}
	}
	for (int i = 0; i < matchPoints.size(); i++) {
		if (!exist[i]) {
			matchPoints[i].first->matchPoint = nullptr;
		}
	}
	
}

void Common::ProjectToCylinder(cv::Mat& src, cv::Mat& dest, float f) {
	int rows = src.rows, cols = src.cols;
	cv::Mat mMapX(rows, cols, CV_32FC1, cv::Scalar(0));
	cv::Mat mMapY(rows, cols, CV_32FC1, cv::Scalar(0));

	for (int y = 0; y < rows; y++) {
		float* ptrX = mMapX.ptr<float>(y);
		float* ptrY = mMapY.ptr<float>(y);
		for (int x = 0; x < cols; x++) {
			int newX = x - cols / 2;
			int newY = y - rows / 2;
			float theta = atanf(newX / f);;
			float castX = (newX / cosf(theta));
			float castY = newY * sqrtf(castX * castX + f * f) / f;
			castX += cols / 2;
			castY += rows / 2;
			if (castX >= 0 && castX < cols && castY >= 0 && castY < rows) {
				ptrX[x] = castX;
				ptrY[x] = castY;
			}
			else {
				// �ɶ¦�
				ptrX[x] = -1;
				ptrY[x] = -1;
			}
		}
	}
	cv::remap(src, dest, mMapX, mMapY, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::imwrite(std::to_string(rand()) + "cylinder.png", dest);
}

cv::Mat Common::FindHomography(const std::vector<FeatureDescriptor>& fPairs, int N) {
	srand(time(NULL));
	std::vector<cv::Point2d> srcPoint, destPoint;
	cv::Mat result(3, 3, CV_64F, (double)0);

	for (int i = 0; i < fPairs.size(); i++) {
		if (fPairs[i].matchPoint != nullptr) {
			srcPoint.push_back(cv::Point2d(fPairs[i].x, fPairs[i].y));
			destPoint.push_back(cv::Point2d(fPairs[i].matchPoint->x, fPairs[i].matchPoint->y));
		}
	}

	int numOfPoints = srcPoint.size();
	double errorNum = INFINITY;
	for (int i = 0; i < numOfPoints; i++) {
		double localError = 0;
		double dx = srcPoint[i].x - destPoint[i].x;
		double dy = srcPoint[i].y - destPoint[i].y;

		for (int j = 0; j < numOfPoints; j++) {
			double firstTerm = srcPoint[i].x - (destPoint[i].x + dx);
			double secondTerm = srcPoint[i].y - (destPoint[i].y + dy);
			localError += firstTerm * firstTerm + secondTerm * secondTerm;
		}
		if (localError < errorNum) {
			errorNum = localError;
			result.at<double>(0, 2) = dx;
			result.at<double>(1, 2) = dy;
		}
	}
	result.at<double>(0, 0) = 1;
	result.at<double>(1, 1) = 1;
	result.at<double>(2, 2) = 1;

	/*
	// RANSAC: N��
	for (int i = 0; i < N; i++) {
		cv::Mat A(8, 9, CV_64F, (double)0);
		cv::Mat h(9, 1, CV_64F, (double)0);
		cv::Mat b(8, 1, CV_64F, (double)0);
		// Fill Matrix A and Matrix b
		for (int k = 0; k < 4; k++) {
			int index = rand() % numOfPoints;
			A.at<double>(2 * k, 0) = srcPoint[index].x;
			A.at<double>(2 * k, 1) = srcPoint[index].y;
			A.at<double>(2 * k, 2) = 1;
			A.at<double>(2 * k, 6) = -srcPoint[index].x * destPoint[index].x;
			A.at<double>(2 * k, 7) = -srcPoint[index].y * destPoint[index].x;
							 
			A.at<double>(2 * k + 1, 3) = srcPoint[index].x;
			A.at<double>(2 * k + 1, 4) = srcPoint[index].y;
			A.at<double>(2 * k + 1, 5) = 1;
			A.at<double>(2 * k + 1, 6) = -srcPoint[index].x * destPoint[index].y;
			A.at<double>(2 * k + 1, 7) = -srcPoint[index].y * destPoint[index].y;

			A.at<double>(2 * k, 8) = -destPoint[index].x;
			A.at<double>(2 * k + 1, 8) = -destPoint[index].y;
							 
			b.at<double>(2 * k, 0) = destPoint[index].x;
			b.at<double>(2 * k + 1, 0) = destPoint[index].y;
		}

		//for (int r = 0; r < 8; r++) {
		//	for (int c = 0; c < 8; c++) {
		//		std::cout << A.at<double>(r, c) << "\t";
		//	}
		//	std::cout << std::endl;
		//}
		// Solve
		cv::Mat At = A.t();
		cv::Mat inverse = (At * A).inv();
		h = inverse * At * b;
		// h normalize
		for (int k = 0; k < 9; k++)
			h.at<double>(k, 0) /= h.at<double>(8, 0);
		// Compute local error
		double localError = 0;
		for (int k = 0; k < numOfPoints; k++) {
			double divider =
				h.at<double>(6, 0) * srcPoint[k].x +
				h.at<double>(7, 0) * srcPoint[k].y +
				h.at<double>(8, 0);
			double firstTerm = destPoint[k].x - 
				(h.at<double>(0, 0) * srcPoint[k].x + h.at<double>(1, 0) * srcPoint[k].y + h.at<double>(2, 0)) / divider;
			double secondTerm = destPoint[k].y - 
				(h.at<double>(3, 0) * srcPoint[k].x + h.at<double>(4, 0) * srcPoint[k].y + h.at<double>(5, 0)) / divider;

			localError += firstTerm * firstTerm + secondTerm * secondTerm;
		}

		if (localError < errorNum) {
			errorNum = localError;
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 3; c++) {
					result.at<double>(r, c) = h.at<double>(r * 3 + c, 0);
				}
			}
			std::cout << "error update" << std::endl;
		}
	}
	*/
	return result;
}
