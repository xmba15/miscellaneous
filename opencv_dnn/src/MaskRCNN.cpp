/**
 * @file    MaskRCNN.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "Constants.hpp"
#include "Utility.hpp"

#include <chrono>

static constexpr float CONF_THRESH = 0.7;  // Confidence threshold
static constexpr float MASK_THRESH = 0.3;  // Mask threshold

static const std::vector<cv::Scalar> MSCOCO_COLORS =
    util::toCvScalarColors(MSCOCO_COLOR_CHART);

// MaskRCNNOuputType->(bboxes, confs, classIds, masks)
using MaskRCNNOuputType =
    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<uint64_t>,
               std::vector<cv::Mat>>;

namespace
{
inline cv::Mat visualizeOneImage(const cv::Mat &frame,
                                 const MaskRCNNOuputType &outputs,
                                 const float maskThreshold,
                                 const std::vector<cv::Scalar> &allColors,
                                 const std::vector<std::string> &allClasses);

MaskRCNNOuputType postprocess(const int width, const int height,
                              const std::vector<cv::Mat> &inferenceOuts,
                              const float confThreshold);

}  // namespace

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cerr
            << "Usage: [apps] [path/to/text/graph] [path/to/model/weight] "
               "[path/to/video]"
            << std::endl;
        return EXIT_FAILURE;
    }

    // mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
    const std::string textGraph = argv[1];

    // mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
    const std::string modelWeights = argv[2];

    const std::string videoPath = argv[3];

    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelWeights, textGraph);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video " << videoPath << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat blob;
    cv::Mat frame;

    int count = 0;
    double totalTime = 0;

    for (;;) {
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // cv::resize(frame, frame, cv::Size(960, 540));

        std::chrono::high_resolution_clock::time_point begin =
            std::chrono::high_resolution_clock::now();

        cv::dnn::blobFromImage(frame, blob, 1.0,
                               cv::Size(frame.cols, frame.rows), cv::Scalar(),
                               true, false);
        net.setInput(blob);
        std::vector<cv::String> outNames(2);
        outNames[0] = "detection_out_final";
        outNames[1] = "detection_masks";
        std::vector<cv::Mat> outs;
        net.forward(outs, outNames);

        auto outputs = ::postprocess(frame.cols, frame.rows, outs, CONF_THRESH);

        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        auto elapsedTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        totalTime += elapsedTime.count();
        count++;

        frame = ::visualizeOneImage(frame, outputs, MASK_THRESH, MSCOCO_COLORS,
                                    MSCOCO_LABELS);

        cv::imshow("output", frame);

        char c = static_cast<char>(cv::waitKey(25));
        if (c == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << totalTime / count << "\n";

    return EXIT_SUCCESS;
}

namespace
{
MaskRCNNOuputType postprocess(const int width, const int height,
                              const std::vector<cv::Mat> &inferenceOuts,
                              const float confThreshold)
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;
    std::vector<cv::Mat> masks;

    cv::Mat outDetections = inferenceOuts[0];
    cv::Mat outMasks = inferenceOuts[1];

    const int numDetections = outDetections.size[2];
    bboxes.reserve(numDetections);
    scores.reserve(numDetections);
    classIndices.reserve(numDetections);
    masks.reserve(numDetections);

    outDetections = outDetections.reshape(1, outDetections.total() / 7);

    for (int i = 0; i < numDetections; ++i) {
        const float *rowPtr = outDetections.ptr<float>(i);
        const float score = rowPtr[2];

        if (score > confThreshold) {
            // Extract the bounding box
            const int classId = static_cast<int>(rowPtr[1]);
            int xmin = static_cast<int>(width * rowPtr[3]);
            int ymin = static_cast<int>(height * rowPtr[4]);
            int xmax = static_cast<int>(width * rowPtr[5]);
            int ymax = static_cast<int>(height * rowPtr[6]);

            xmin = std::max(0, std::min(xmin, width - 1));
            ymin = std::max(0, std::min(ymin, height - 1));
            xmax = std::max(0, std::min(xmax, width - 1));
            ymax = std::max(0, std::min(ymax, height - 1));

            cv::Rect bbox =
                cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);

            cv::Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F,
                               outMasks.ptr<float>(i, classId));

            bboxes.emplace_back(bbox);
            scores.emplace_back(score);
            classIndices.emplace_back(classId);
            masks.emplace_back(objectMask);
        }
    }

    return std::make_tuple(bboxes, scores, classIndices, masks);
}

cv::Mat visualizeOneImage(const cv::Mat &frame,
                          const MaskRCNNOuputType &outputs,
                          const float maskThreshold,
                          const std::vector<cv::Scalar> &allColors,
                          const std::vector<std::string> &allClasses)
{
    cv::Mat result = frame.clone();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;
    std::vector<cv::Mat> masks;

    std::tie(bboxes, scores, classIndices, masks) = outputs;

    for (size_t i = 0; i < bboxes.size(); ++i) {
        const cv::Rect &curBbox = bboxes[i];
        const float curScore = scores[i];
        const uint64_t curClassIdx = classIndices[i];
        cv::Mat &curMask = masks[i];
        const cv::Scalar &curColor = allColors[curClassIdx];

        cv::rectangle(result, curBbox, curColor, 2);

        const std::string curLabel =
            allClasses[curClassIdx] + ":" + cv::format("%.2f", curScore);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(curLabel, cv::FONT_HERSHEY_COMPLEX,
                                             0.35, 1, &baseLine);
        cv::rectangle(
            result, cv::Point(curBbox.x, curBbox.y),
            cv::Point(curBbox.x + labelSize.width,
                      curBbox.y + static_cast<int>(1.3 * labelSize.height)),
            curColor, -1);

        cv::putText(result, curLabel,
                    cv::Point(curBbox.x, curBbox.y + labelSize.height),
                    cv::FONT_HERSHEY_COMPLEX, 0.35, cv::Scalar(255, 255, 255));

        cv::Mat another;
        cv::resize(curMask, curMask, cv::Size(curBbox.width, curBbox.height));

        cv::Mat finalMask = (curMask > maskThreshold);

        cv::Mat coloredRoi = (0.3 * curColor + 0.7 * result(curBbox));

        coloredRoi.convertTo(coloredRoi, CV_8UC3);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        finalMask.convertTo(finalMask, CV_8U);
        cv::findContours(finalMask, contours, hierarchy, cv::RETR_CCOMP,
                         cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(coloredRoi, contours, -1, curColor, 5, cv::LINE_8,
                         hierarchy, 100);
        coloredRoi.copyTo(result(curBbox), finalMask);
    }

    return result;
}
}  // namespace
