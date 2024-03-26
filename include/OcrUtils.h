#ifndef __OCR_UTILS_H__
#define __OCR_UTILS_H__

#include <opencv2/core.hpp>
#include "OcrStruct.h"
#include <onnxruntime_cxx_api.h>
#include <numeric>
#include <sys/stat.h>

const double PI = 3.14159265358979323846;

template<typename T, typename... Ts>
static std::unique_ptr<T> makeUnique(Ts &&... params) {
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

template<typename T>
static double getMean(std::vector<T> &input) {
    auto sum = accumulate(input.begin(), input.end(), 0.0);
    return sum / input.size();
}

template<typename T>
static double getStdev(std::vector<T> &input, double mean) {
    if (input.size() <= 1) return 0;
    double accum = 0.0;
    for_each(input.begin(), input.end(), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });
    double stdev = sqrt(accum / (input.size() - 1));
    return stdev;
}

template<class T>
inline T clamp(T x, T min, T max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

double getCurrentTime();

inline bool isFileExists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::wstring strToWstr(std::string str);

ScaleParam getScaleParam(cv::Mat &src, const float scale);

ScaleParam getScaleParam(cv::Mat &src, const int targetSize);

std::vector<cv::Point2f> getBox(const cv::RotatedRect &rect);

int getThickness(cv::Mat &boxImg);

void drawTextBox(cv::Mat &boxImg, cv::RotatedRect &rect, int thickness);

void drawTextBox(cv::Mat &boxImg, const std::vector<cv::Point> &box, int thickness);

void add_boxes_center(std::vector<TextBox>& textBoxes);

void tilt_correction(cv::Mat &src, std::vector<TextBox>& textBoxes, double line_angle, double angle_threshold=2);

void rotate_points(cv::Mat& src, std::vector<TextBox>& textBoxes, double line_angle, int mode = 1);

double text_line_angle(std::vector<TextBox>& textBoxes);

bool compare_arc_length(const TextBox& box1, const TextBox& box2);

bool compare_box_position(const TextBox& box1, const TextBox& box2);

void sort_boxes(std::vector<TextBox>& textBoxes);

double compute_box_angle(const TextBox& text_box);

void drawTextBoxes(cv::Mat &boxImg, std::vector<TextBox> &textBoxes, int thickness);

cv::Mat matRotateClockWise180(cv::Mat src);

cv::Mat matRotateClockWise90(cv::Mat src);

cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<cv::Point> box);

cv::Mat adjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight);

std::vector<cv::Point2f> getMinBoxes(const cv::RotatedRect &boxRect, float &maxSideLen);

float boxScoreFast(const std::vector<cv::Point2f> &boxes, const cv::Mat &pred);

cv::RotatedRect unClip(std::vector<cv::Point2f> box, float unClipRatio);

std::vector<float> substractMeanNormalize(cv::Mat &src, const float *meanVals, const float *normVals);

std::vector<int> getAngleIndexes(std::vector<Angle> &angles);

std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session *session);

std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session *session);

void saveImg(cv::Mat &img, const char *imgPath);

std::string getSrcImgFilePath(const char *path, const char *imgName);

std::string getResultTxtFilePath(const char *path, const char *imgName);

std::string getResultImgFilePath(const char *path, const char *imgName);

std::string getDebugImgFilePath(const char *path, const char *imgName, size_t i, const char *tag);

#endif //__OCR_UTILS_H__
