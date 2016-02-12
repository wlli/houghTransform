//
//  main.cpp
//  Hough
//
//  Copyright © 2016 wanlong. All rights reserved.
//

#include <iostream>
#include "hough.hpp"

using namespace cv;

int sThreshold = 100;
int sThetaMin = 0;
int sThetaMax = 179;
int sExtendingLine = 0;
const char* sInputFilePath = "";
const char* sOutputFilePath = "";

void doHoughTransformAndShowResult();

void onTrackbarChanged(int newVal,void* data)
{
    int trackbarCode = *((int*)data);
    switch (trackbarCode) {
        case 0:
            sThreshold = newVal;
            break;
        case 1:
            sThetaMin = newVal;
            break;
        case 2:
            sThetaMax = newVal;
            break;
        case 3:
            sExtendingLine = newVal;
            break;
        default:
            assert(false);
            return;
    }
    
    doHoughTransformAndShowResult();
}

void displayResult(const char* resultFilePath, int thresholdVal, int thetaLowVal, int thetaHighVal, int isExtendingLine)
{
    cv::Mat mat = cv::imread(resultFilePath);
    
    if (!mat.data)
    {
        std::cout << "Image Not Found.." << std::endl;
        return;
    }
    
    std::string windowName = "output_image";
    int trackbar1 = 0;
    int trackbar2 = 1;
    int trackbar3 = 2;
    int trackbar4 = 3;
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold:", windowName, &thresholdVal, 1000, onTrackbarChanged, &trackbar1);
    cv::createTrackbar("Theta-Low:", windowName, &thetaLowVal, 179, onTrackbarChanged, &trackbar2);
    cv::createTrackbar("Theta-High:", windowName, &thetaHighVal, 179, onTrackbarChanged, &trackbar3);
    cv::createTrackbar("Mode:", windowName, &isExtendingLine, 1, onTrackbarChanged, &trackbar4);
    
    imshow(windowName, mat);
    
    waitKey(0);
    destroyAllWindows();
}

void doHoughTransformAndShowResult()
{
    Hough hough;
    if (sThreshold > 0 && hough.execute(sInputFilePath, sOutputFilePath, sExtendingLine, sThreshold, sThetaMin, sThetaMax))
    {
        displayResult(sOutputFilePath, sThreshold, sThetaMin, sThetaMax, sExtendingLine);
    }
    else
    {
        displayResult(sInputFilePath, sThreshold, sThetaMin, sThetaMax, sExtendingLine);
    }
}

int main(int argc, const char * argv[])
{
    std::cout << "Using opencv version " << CV_VERSION <<"\n";
    
    sInputFilePath = argv[1];
    sOutputFilePath = argv[2];
    doHoughTransformAndShowResult();

    return 0;
}

