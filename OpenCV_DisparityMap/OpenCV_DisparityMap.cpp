// OpenCV_DisparityMap.cpp: определяет точку входа для приложения.
//

#include "OpenCV_DisparityMap.h"
#include<opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

using namespace std;
using namespace cv;
using namespace ximgproc;

Mat readImage(string name)
{
	Mat im11 = imread("../../../../img/" + name, IMREAD_GRAYSCALE);
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
    //cvtColor(src, src_gray, COLOR_BGR2GRAY);
    src_gray = src;
    int blockSize = 4;
    int SobelSize = 5;
    double k = 0.04;
    int thresh = 150;
    

    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, SobelSize, k);
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

template<typename tp>
void printStdVec(vector<tp> errors, string header)
{
    if (header.empty())
        cout << "std vec:  ";
    else
        cout << header << ":  ";
    for (int i = 0; i < errors.size(); i++)
    {
        cout << errors[i] << "  ";
    }
    cout << endl;
}

Mat disparity(const Mat& left, const Mat& right)
{
    Mat left_for_matcher, right_for_matcher;

    int max_disp = 100;
    max_disp /= 2;
    if (max_disp % 16 != 0)
        max_disp += 16 - (max_disp % 16);
    resize(left, left_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);
    resize(right, right_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR_EXACT);


    Mat left_disp, right_disp;
    Ptr<StereoBM> left_matcher = StereoBM::create(max_disp);
    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilter(left_matcher);
    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

    left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
    right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

    Mat filtered_disp, solved_disp, solved_filtered_disp;

    wls_filter->setLambda(2.0);
    wls_filter->setSigmaColor(1.0);

    wls_filter->filter(left_disp, left, filtered_disp, right_disp);

    double vis_mult = 1.0;
    Mat raw_disp_vis;
    getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    //  namedWindow("raw disparity", WINDOW_AUTOSIZE);
    //  imshow("raw disparity", raw_disp_vis);

    Mat filtered_disp_vis;
    getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    imshow("filtered disparity", filtered_disp_vis);

    if (!solved_disp.empty())
    {
        Mat solved_disp_vis;
        getDisparityVis(solved_disp, solved_disp_vis, vis_mult);
        namedWindow("solved disparity", WINDOW_AUTOSIZE);
        imshow("solved disparity", solved_disp_vis);
        Mat solved_filtered_disp_vis;
        getDisparityVis(solved_filtered_disp, solved_filtered_disp_vis, vis_mult);
        namedWindow("solved wls disparity", WINDOW_AUTOSIZE);
        imshow("solved wls disparity", solved_filtered_disp_vis);
    }
    return raw_disp_vis;
}

int main()
{
    
	Mat im1 = readImage("31.jpg");
	Mat im2 = readImage("32.jpg");
    
    vector<Point2f> mass1, mass2;
    mass1 = cornerHarris_myShell(im1);
    mass2 = cornerHarris_myShell(im2);
    
    system("cls");
    cout << "Size of mass1 = " << mass1.size() << ",  Size of mass2 = " << mass2.size() << endl;
    
    for (int i = 0; i < mass1.size(); i++)
    {
        Point p = mass1[i];
        circle(im1, p, 5, Scalar(0), 2, 8, 0);
    }

    for (int i = 0; i < mass2.size(); i++)
    {
        Point p = mass2[i];
        circle(im2, p, 5, Scalar(0), 2, 8, 0);
    }
    

    if (mass1.size() < 8 || mass2.size() < 8)
    {
        cout << "Vector of features too low" << endl;
        char c;
        cin >> c;
        exit(0);
    }

    vector<float> errors;
    Mat status;
    calcOpticalFlowPyrLK(im1, im2, mass1, mass2, status, errors, Size(21,21), 3);
    cout << "Size of mass1 = " << mass1.size() << ",  Size of mass2 = " << mass2.size() << endl;

    for (int i = 0; i < mass1.size(); i++)
    {
        Point p = mass1[i];
        circle(im1, p, 5, Scalar(0), 2, 8, 0);
    }


    for (int i = 0; i < mass2.size(); i++)
    {
        Point p = mass2[i];
        circle(im2, p, 5, Scalar(0), 2, 8, 0);
    }
    //  imshow("im1", im1);
    //  imshow("im2", im2);


    Mat F = findFundamentalMat(mass1, mass2, FM_RANSAC,3,0.99);
    
    Mat H1, H2;
    stereoRectifyUncalibrated(mass1, mass2, F, im1.size(),H1,H2,5);

    Mat newIm1, newIm2, newIm3;
    
    warpPerspective(im1, newIm1, H1, im1.size(), INTER_LINEAR|WARP_INVERSE_MAP, BORDER_REPLICATE);
    warpPerspective(im2, newIm2, H2, im1.size(), INTER_LINEAR|WARP_INVERSE_MAP, BORDER_REPLICATE);
    warpPerspective(im2, newIm3, H2*H1.inv(), im1.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE);

    disparity(newIm1, newIm2);
    
    //  imshow("im1", im1);
    //  imshow("im2", im2);
    imshow("newIm1", newIm1);
    imshow("newIm2", newIm2);
    //  imshow("newIm3", newIm3);

    waitKey();
	system("pause");
	return 0;
}
