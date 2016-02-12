//
//  hough.hpp
//  Hough
//
//  Copyright © 2016 wanlong. All rights reserved.
//

#ifndef hough_hpp
#define hough_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

#endif /* hough_hpp */
using namespace cv;
class Hough
{
public:
    bool execute(const char* filePath, const char* outputFilePath, bool bMarkExtend,int threshold = 100, int thetaMin = 0, int  thetaMax = 179);
    Hough();
    ~Hough(){}
    
private:
    bool prepareData(Mat& img_orig, Mat& img_edge, const char* filePath);
    bool transform(cv::Mat& img_edge, std::vector<std::vector<int>>& accumulator);
    void backMapping(Mat& img, Mat& imgOut, std::vector<std::vector<int>>& accumulator, bool bMarkExtend);
    void findLocalMaxima(std::vector<std::vector<int>>& vec2d, int radiusX, int radiusY);
    void displayMat(Mat& mat);
    
private:
    int mShowWindowIndex;
    int mWidth;
    int mHeight;
    int mDiagonal;
    double sinVals[180];
    double cosVals[180];
    int mThetaMin;
    int mThetaMax;
    int mThreshold;
};
