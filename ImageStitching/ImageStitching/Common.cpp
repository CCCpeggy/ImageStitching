#include "Common.h"
#include <cmath>

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

