// OpenCV_DisparityMap.cpp: определяет точку входа для приложения.
//

#include "OpenCV_DisparityMap.h"
#include<opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

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
    int SobelSize = 15;
    double k = 0.04;
    int thresh = 180;

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

int main()
{
    
	Mat im1 = readImage("51.jpg");
	Mat im2 = readImage("52.jpg");
    
    vector<Point2f> mass1, mass2;
    mass1 = cornerHarris_myShell(im1);
    mass2 = cornerHarris_myShell(im2);
    
    system("cls");
    //  cout <<"mass1 = "<< mass1 << endl;
    //  cout <<"mass2 = "<< mass2 << endl;

    cout << "Size of mass1 = " << mass1.size() << ",  Size of mass2 = " << mass2.size() << endl;
    
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
    //  cout << "status " << status.t() << endl;
    //  printStdVec(errors, "errors");
    
    

    
    vector<Point2f> mass1F, mass2F;
    
    mass1F = mass1;
    mass2F = mass2;

    //  for (int i = 0; i < errors.size(); i++)
    //      if (errors[i] < 10)
    //      {
    //          mass1F.push_back(mass1[i]);
    //          mass2F.push_back(mass2[i]);
    //      }
    //  
    //  cout << "Mass1F = " << mass1F << endl;
    //  cout << "Mass2F = " << mass2F << endl;
    


    Mat F = findFundamentalMat(mass1F, mass2F, FM_RANSAC,3,0.99);
    

    
    cout << "Im1 size = " << im1.size() << "  Im2 size = " << im2.size() << endl;
    Mat H1, H2;
    stereoRectifyUncalibrated(mass1F, mass2F, F, im1.size(),H1,H2,5);

    Mat newIm1, newIm2, newIm3;
    
    warpPerspective(im1, newIm1, H1, im1.size(), INTER_LINEAR|WARP_INVERSE_MAP, BORDER_REPLICATE);
    warpPerspective(im2, newIm2, H2, im1.size(), INTER_LINEAR|WARP_INVERSE_MAP, BORDER_REPLICATE);

    warpPerspective(im2, newIm3, H2*H1.inv(), im1.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE);

    imshow("im1", im1);
    imshow("im2", im2);
    imshow("newIm1", newIm1);
    imshow("newIm2", newIm2);
    imshow("newIm3", newIm3);

    waitKey();
	system("pause");
	return 0;
}
