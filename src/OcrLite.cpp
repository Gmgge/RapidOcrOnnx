#include "OcrLite.h"
#include "OcrUtils.h"
#include <stdarg.h> //windows&linux

OcrLite::OcrLite() {}

OcrLite::~OcrLite() {
    if (isOutputResultTxt) {
        fclose(resultTxt);
    }
}

void OcrLite::setNumThread(int numOfThread) {
    dbNet.setNumThread(numOfThread);
    angleNet.setNumThread(numOfThread);
    crnnNet.setNumThread(numOfThread);
}

void OcrLite::initLogger(bool isConsole, bool isPartImg, bool isResultImg) {
    isOutputConsole = isConsole;
    isOutputPartImg = isPartImg;
    isOutputResultImg = isResultImg;
}

void OcrLite::enableResultTxt(const char *path, const char *imgName) {
    isOutputResultTxt = true;
    std::string resultTxtPath = getResultTxtFilePath(path, imgName);
    printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(resultTxtPath.c_str(), "w");
}

void OcrLite::setGpuIndex(int gpuIndex) {
    dbNet.setGpuIndex(gpuIndex);
    angleNet.setGpuIndex(-1);
    crnnNet.setGpuIndex(gpuIndex);
}

bool OcrLite::initModels(const std::string &detPath, const std::string &clsPath,
                         const std::string &recPath, const std::string &keysPath) {
    Logger("=====Init Models=====\n");
    Logger("--- Init DbNet ---\n");
    dbNet.initModel(detPath);

    Logger("--- Init AngleNet ---\n");
    angleNet.initModel(clsPath);

    Logger("--- Init CrnnNet ---\n");
    crnnNet.initModel(recPath, keysPath);

    Logger("Init Models Success!\n");
    return true;
}

void OcrLite::Logger(const char *format, ...) {
    if (!(isOutputConsole || isOutputResultTxt)) return;
    char *buffer = (char *) malloc(8192);
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    if (isOutputConsole) printf("%s", buffer);
    if (isOutputResultTxt) fprintf(resultTxt, "%s", buffer);
    free(buffer);
}

cv::Mat makePadding(cv::Mat &src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

OcrResult OcrLite::detect(const char *path, const char *imgName,
                          const int padding, const int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    std::string imgFile = getSrcImgFilePath(path, imgName);

    cv::Mat originSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(path, imgName, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

OcrResult OcrLite::detect(const cv::Mat &mat, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
                          float unClipRatio, bool doAngle, bool mostAngle) {
    cv::Mat originSrc = mat;
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(NULL, NULL, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

std::vector<cv::Mat> OcrLite::getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes,
                                            const char *path, const char *imgName) {
    std::vector<cv::Mat> partImages;
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
        //OutPut DebugImg
        if (isOutputPartImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-part-");
            saveImg(partImg, debugImgFile.c_str());
        }
    }
    return partImages;
}

OcrResult OcrLite::detect(const char *path, const char *imgName,
                          cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {

    //cv::Mat textBoxPaddingImg = src.clone();
    //int thickness = getThickness(src);

    Logger("=====Start OCR=====\n");
    Logger("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n", scale.srcWidth, scale.srcHeight,
           scale.dstWidth, scale.dstHeight,
           scale.ratioWidth, scale.ratioHeight);

    //Logger("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)\n", dbNetTime);

    //for (size_t i = 0; i < textBoxes.size(); ++i) {
    //    Logger("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
    //           textBoxes[i].score,
    //           textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
    //           textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
    //           textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
    //           textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    //}

    //Logger("---------- step: drawTextBoxes ----------\n");
    //drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    // Text line angle correction and sort
    double line_rotate = text_line_angle(textBoxes);
    Logger("Text line rotate(%f)\n", line_rotate);
    add_boxes_center(textBoxes);
    tilt_correction(src, textBoxes, line_rotate);
    sort_boxes(textBoxes);

    //---------- getPartImages ----------
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

    //Logger("---------- step: angleNet getAngles ----------\n");
    std::vector<Angle> angles;
    angles = angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

    //Log Angles
    //for (size_t i = 0; i < angles.size(); ++i) {
    //    Logger("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
    //}

    //Rotate partImgs
    int flip_count = 0;
    for (size_t i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 1) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
            flip_count += 1;
        }
    }
    bool flip_sign = (flip_count / (partImages.size() + 0.00001)) > 0.6;
    Logger("Flip sign (%d)\n", flip_sign);
    
    //Text image flip correction
    if (flip_sign){
        std::reverse(partImages.begin(), partImages.end());
        std::reverse(textBoxes.begin(), textBoxes.end());
    }

    //Logger("---------- step: crnnNet getTextLine ----------\n");
    std::vector<TextLine> textLines = crnnNet.getTextLines(partImages, path, imgName);
    
    //Log TextLines
    //for (size_t i = 0; i < textLines.size(); ++i) {
    //    Logger("textLine[%d](%s)\n", i, textLines[i].text.c_str());
    //    std::ostringstream txtScores;
    //    for (size_t s = 0; s < textLines[i].charScores.size(); ++s) {
    //        if (s == 0) {
    //            txtScores << textLines[i].charScores[s];
    //        } else {
    //            txtScores << " ," << textLines[i].charScores[s];
    //        }
    //    }
    //    Logger("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
    //    Logger("crnnTime[%d](%fms)\n", i, textLines[i].time);
    //}

    // Filter text line
    std::vector<TextBox> filter_boxes;
    std::vector<TextLine> filter_lines;
    std::vector<Angle> filter_angles;
    for (size_t i = 0; i < textLines.size(); ++i) {
        double score_mean = 0.0;
        double score_sum = 0.0;
        for (size_t s = 0; s < textLines[i].charScores.size(); ++s) {
            score_sum += textLines[i].charScores[s];
        }
        score_mean = score_sum / textLines[i].charScores.size();

        if (score_mean > 0.5) {
            
            filter_boxes.push_back(textBoxes[i]);
            filter_lines.push_back(textLines[i]);
            filter_angles.push_back(angles[i]);
        }
    }

    std::vector<TextBlock> textBlocks;
    for (size_t i = 0; i < filter_lines.size(); ++i) {
        std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        int padding = originRect.x;//padding conversion
        boxPoint[0] = cv::Point(filter_boxes[i].boxPoint[0].x - padding, filter_boxes[i].boxPoint[0].y - padding);
        boxPoint[1] = cv::Point(filter_boxes[i].boxPoint[1].x - padding, filter_boxes[i].boxPoint[1].y - padding);
        boxPoint[2] = cv::Point(filter_boxes[i].boxPoint[2].x - padding, filter_boxes[i].boxPoint[2].y - padding);
        boxPoint[3] = cv::Point(filter_boxes[i].boxPoint[3].x - padding, filter_boxes[i].boxPoint[3].y - padding);
        TextBlock textBlock{ boxPoint, filter_boxes[i].score, filter_angles[i].index, filter_angles[i].score,
                            filter_angles[i].time, filter_lines[i].text, filter_lines[i].charScores, filter_lines[i].time,
                            filter_angles[i].time + filter_lines[i].time };
        textBlocks.emplace_back(textBlock);
    }

    //cropped to original size
    //cv::Mat textBoxImg;

    //if (originRect.x > 0 && originRect.y > 0) {
    //    textBoxPaddingImg(originRect).copyTo(textBoxImg);
    //} else {
    //    textBoxImg = textBoxPaddingImg;
    //}

    ////Save result.jpg
    //if (isOutputResultImg) {
    //    std::string resultImgFile = getResultImgFilePath(path, imgName);
    //    imwrite(resultImgFile, textBoxImg);
    //}

    // Text line layout recovery
    std::string strRes = "";
    if (!filter_lines.empty())
    {
        TextBox pre_box = filter_boxes[0];
        strRes.append(filter_lines[0].text);
        for (size_t i = 1; i < filter_lines.size(); ++i) {
            double pre_box_center_x = flip_sign ? -pre_box.boxPoint[4].x : pre_box.boxPoint[4].x;
            double box_center_x = flip_sign ? -filter_boxes[i].boxPoint[4].x : filter_boxes[i].boxPoint[4].x;
            if ((abs(pre_box.boxPoint[4].y - filter_boxes[i].boxPoint[4].y) < 10) && (pre_box_center_x < box_center_x)) {
                strRes.append(" ");
                strRes.append(filter_lines[i].text);
            }
            else {
                strRes.append("\n");
                strRes.append(filter_lines[i].text);
            }
            pre_box = filter_boxes[i];
        }
    }

    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End OCR=====\n");
    Logger("FullDetectTime(%fms)\n", fullTime);

    return OcrResult{dbNetTime, textBlocks, fullTime, strRes};
}