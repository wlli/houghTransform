//
//  hough.cpp
//  Hough
//
//  Copyright © 2016 wanlong. All rights reserved.
//


#include "hough.hpp"
#include <math.h>
Hough::Hough()
:mShowWindowIndex(0),mWidth(0),mHeight(0),mDiagonal(0)
{
    //prepare sin(), cos() calculations
    const double D2R = M_PI / 180;
    for (int d = 0; d < 180; ++d)
    {
        sinVals[d] = sin(d * D2R);
        cosVals[d] = cos(d * D2R);
    }
}

//output a Mat of edge image
bool Hough::prepareData(cv::Mat& img_orig, cv::Mat& img_edge, const char* filePath)
{
    img_orig = imread(filePath, CV_LOAD_IMAGE_COLOR);
    if (!img_orig.data)
        return false;
    
    Mat img_blur;
    blur(img_orig, img_blur, Size(5 ,5));
    if (!img_blur.data)
        return false;
    
    Canny(img_blur, img_edge, 100, 150, 3);
    if (!img_edge.data)
        return false;
    
    return true;
}

//accumulator: (theta, rho) where 0<=theta<=180, and -1*diagonal<=rho<=diagonal
//              and diagonal=sqrt(width^2+height^2)
bool Hough::transform(cv::Mat& img_edge, std::vector<std::vector<int>>& accumulator)
{
    uchar* data = img_edge.data;
    if (!data)
        return false;
    
    //r = x*cos + y*sin
    for (int y = 0; y < mHeight; ++y)
    {
        for (int x = 0; x < mWidth; ++x)
        {
            if (data[y * mWidth + x] > 250)
            {
                for (int k = 0; k < mThetaMax - mThetaMin + 1; ++k)
                {
                    int rho = (int)round(x * cosVals[k + mThetaMin] + y * sinVals[k + mThetaMin]);
                    rho += mDiagonal; //shift upwards
                    ++accumulator[rho][k];
                }
            }
        }
    }
    
    // --------Check data integrity---------
    int count = 0;
    int total = 0;
    int over100 = 0;
    int max = 0;
    size_t cols = accumulator[0].size();
    for (auto itr = accumulator.begin(); itr != accumulator.end(); itr++)
    {
        assert(cols == itr->size());
        for (auto itr2 = itr->begin(); itr2 != itr->end(); itr2++)
        {
            ++count;
            total += *itr2;
            if (*itr2 > 100) ++over100;
            max = *itr2 > max ? *itr2 : max;
        }
    }
    //Mat matHough(accumulator, true);
    cv::Mat matHough((int)accumulator.size(), (int)accumulator[0].size(), CV_8UC1);
    
    //risk of overflow
    for (size_t i = 0; i < accumulator.size(); i++)
    {
        for (size_t j = 0; j < accumulator[0].size(); j++)
        {
            matHough.at<uchar>((int)i,(int)j) = /*using white background after normalization*/
                255 - (double)accumulator[i][j] / max * 255;
        }
    }
    //
    //displayMat(matHough);
    //----------End check---------------
    
    return true;
}

void Hough::backMapping(cv::Mat& img, cv::Mat& imgOut, std::vector<std::vector<int>>& accumulator, bool bMarkExtend)
{
    //convert for color marking
    cv::cvtColor(img, imgOut, CV_GRAY2BGR);
    //img.copyTo(imgOut);
    
    findLocalMaxima(accumulator, 10, 20);
    
    if (!bMarkExtend)
    {
        //mapping back to points in image
        std::vector<std::pair<int, int>> accus; //<r, t>
        int r = 0;
        for (auto itr = accumulator.begin(); itr != accumulator.end(); itr++, r++)
        {
            int t = mThetaMin;
            for (auto itr2 = itr->begin(); itr2 != itr->end(); itr2++, t++)
            {
                if (*itr2 > mThreshold)
                {
                    accus.push_back(std::pair<int, int>(r - mDiagonal, t));
                }
            }
        }
        
        uchar* data = img.data;
        for (int y = 0; y < mHeight; ++y)
        {
            for (int x = 0; x < mWidth; ++x)
            {
                if (data[y * mWidth + x] > 250)
                {
                    for (auto itr = accus.begin(); itr != accus.end(); itr++)
                    {
                        if (itr->first == (int)round(x * cosVals[itr->second] + y * sinVals[itr->second]))
                        {
                            cv::circle(imgOut, cv::Point(x, y), 1, Scalar(0,255,0));
                        }
                    }
                }
            }
        }
    }
    else
    {
        //drawing lines
        int r = 0;
        for (auto itr = accumulator.begin(); itr != accumulator.end(); itr++, r++)
        {
            int t = mThetaMin;
            for (auto itr2 = itr->begin(); itr2 != itr->end(); itr2++, t++)
            {
                if (*itr2 > mThreshold)
                {
                    double cos_t = cosVals[t], sin_t = sinVals[t];
                    double x0 = (r - mDiagonal) * cos_t, y0 = (r - mDiagonal) * sin_t;
                    double alpha = 5000;
                    
                    Point pt1(cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t));
                    Point pt2(cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t));
                    
                    line(imgOut, pt1, pt2, Scalar(0,255,0), 1, LINE_AA);
                }
            }
        }
    }
}

void Hough::findLocalMaxima(std::vector<std::vector<int>>& vec2d, int radiusX, int radiusY)
{
    int h = (int)vec2d.size();
    for (int y = 0; y < h; y += radiusY)
    {
        int w = (int)vec2d[y].size();
        for (int x = 0; x < w; x += radiusX)
        {
            int maxVal = 0;
            int maxX = 0;
            int maxY = 0;
            for (int dy = -1 * radiusY; dy <= radiusY; ++dy)
            {
                for (int dx = -1 * radiusX; dx <= radiusX; ++dx)
                {
                    if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < w)
                    {
                        //std::string s = "(" + std::to_string(y+dy) + "," + std::to_string(x+dx) +")";
                        //std::cout << s <<std::endl;
                        if (vec2d[y + dy][x + dx] > maxVal)
                        {
                            vec2d[maxY][maxX] = -1;
                            maxY = y + dy;
                            maxX = x + dx;
                            maxVal = vec2d[y + dy][x + dx];
                        }
                        else
                        {
                            vec2d[y + dy][x + dx] = -1;
                        }
                    }
                }
            }
        }
    }
    
    // --------Check data integrity---------
    int count = 0;
    int total = 0;
    int over100 = 0;
    int max = 0;
    size_t cols = vec2d[0].size();
    for (auto itr = vec2d.begin(); itr != vec2d.end(); itr++)
    {
        assert(cols == itr->size());
        for (auto itr2 = itr->begin(); itr2 != itr->end(); itr2++)
        {
            ++count;
            total += *itr2;
            if (*itr2 > 100) ++over100;
            max = *itr2 > max ? *itr2 : max;
        }
    }

}

bool Hough::execute(const char* filePath, const char* outputFilePath,  bool bMarkExtend, int threshold, int thetaMin, int  thetaMax)
{
    if (thetaMin < 0 || thetaMax > 179 || thetaMax < thetaMin)
        return false;
    
    mThetaMax = thetaMax;
    mThetaMin = thetaMin;
    mThreshold = threshold;
    
    Mat img_orig, img_edge;
    if (!prepareData(img_orig, img_edge, filePath))
        return false;
    
    mWidth = img_edge.cols;
    mHeight = img_edge.rows;
    mDiagonal = ceil(sqrt(mWidth * mWidth + mHeight * mHeight));
    
    //init an accumulator
    std::vector<std::vector<int>> accu(2 * mDiagonal);
    for (auto itr = accu.begin(); itr != accu.end(); itr++)
    {
        *itr = std::vector<int>(mThetaMax - mThetaMin + 1);
        for (auto itr2 = itr->begin(); itr2 != itr->end(); itr2++)
        {
            *itr2 = 0;
        }
    }
    if (!transform(img_edge, accu))
        return false;
    
    //displayMat(img_edge);
    cv::Mat img_marked;
    backMapping(img_edge, img_marked, accu, bMarkExtend);
    //displayMat(img_marked);
    
    cv::imwrite(outputFilePath, img_marked);

    return true;
}


//show Mat as image in a window
void Hough::displayMat(cv::Mat &mat)
{
    std::string windowName = "image_" + std::to_string(mShowWindowIndex++);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    imshow(windowName, mat);
}
