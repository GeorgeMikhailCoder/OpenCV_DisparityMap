// OpenCV_DisparityMap.cpp: определяет точку входа для приложения.
//

#include "OpenCV_DisparityMap.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat readImage(string name)
{
	Mat im11 = imread("../../../../img/" + name, IMREAD_COLOR);
	if (im11.empty())
	{
		cout << "Can't read image" << endl;
		char c;
		cin >> c;
		exit(0);
	}
	return im11;
}

vector<Point2f> cornerHarris_myShell(Mat src)
{
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    int blockSize = 3;
    int apertureSize = 7;
    double k = 0.07;
    int thresh = 180;

    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); // привели к диапазону 0..255 float
    convertScaleAbs(dst_norm, dst_norm_scaled); // привели к CV_8U (saturate_cast)

    vector<Point2f> mass;

    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
                circle(src, Point(j, i), 5, Scalar(0,0,200), 2, 8, 0);
                mass.push_back(Point2f(i, j));
            }
        }
    }
    //  imshow("Corners detected", dst_norm_scaled);
    //  imshow("Corners detected 2", src);
    waitKey();
    return mass;
}

int main()
{
	Mat im1 = readImage("11.jpg");
	Mat im2 = readImage("12.jpg");

    vector<Point2f> mass1, mass2;
    mass1 = cornerHarris_myShell(im1);
    mass2 = cornerHarris_myShell(im2);
    cout << mass1 << endl;
    cout << mass2 << endl;
    mass1 = mass2;
    if (mass1.size() < 8 || mass2.size() < 8)
    {
        cout << "Vector of features too low" << endl;
        char c;
        cin >> c;
        exit(0);
    }

    Mat F = findFundamentalMat(mass1, mass2, FM_RANSAC);
    cout << F << endl;

    Mat newIm1, newIm2;
    stereoRectifyUncalibrated(mass1, mass2, F, im1.size(),newIm1,newIm2);

    Mat dist;
    absdiff(newIm2, newIm1, dist);
    imshow(dist);
    waitKey();
	system("pause");
	return 0;
}
