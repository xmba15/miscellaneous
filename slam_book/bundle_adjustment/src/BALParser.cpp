/**
 * @file    BALParser.cpp
 *
 * @author  btran
 *
 */

#include "BALParser.hpp"

namespace
{
std::vector<std::string> splitByDelim(const std::string& s, const char delimiter)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }
    return tokens;
}

std::vector<std::string> parseMetaDataFile(const std::string& metaDataFilePath)
{
    std::ifstream inFile;
    inFile.open(metaDataFilePath);

    if (!inFile) {
        throw std::runtime_error("unable to open " + metaDataFilePath);
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();

    return splitByDelim(buffer.str(), '\n');
}
}  // namespace

namespace _cv
{
BALParser::BALParser(const std::string& dataFile)
{
    auto allLines = parseMetaDataFile(dataFile);
    auto firstLineElems = splitByDelim(allLines.front(), ' ');
    numCams = std::atoi(firstLineElems[0].c_str());
    numPoints = std::atoi(firstLineElems[1].c_str());
    numObservations = std::atoi(firstLineElems[2].c_str());

    int totalLines = numObservations + 9 * numCams + 3 * numPoints + 1;
    if (totalLines != allLines.size()) {
        throw std::runtime_error("invalid data file");
    }

    camIndices.resize(numObservations);
    pointIndices.resize(numObservations);
    observations.resize(numObservations);

    camPoses.resize(numCams);
    camIntrinsics.resize(numCams, std::vector<float>(3, 0));  // focal length, k1, k2 each

    point3Ds.resize(numPoints);

    // observations
    int curStartLine = 1;
    for (int i = 0; i < numObservations; ++i) {
        std::stringstream ss(allLines[i + curStartLine]);
        std::string buff;
        std::vector<std::string> curLineElems;
        while (ss >> buff) {
            curLineElems.emplace_back(buff);
        }

        camIndices[i] = std::atoi(curLineElems[0].c_str());
        pointIndices[i] = std::atoi(curLineElems[1].c_str());
        observations.emplace_back(cv::Vec2d(std::atof(curLineElems[2].c_str()), std::atof(curLineElems[3].c_str())));
    }

    // camera parameters
    curStartLine += numObservations;

    for (int i = 0; i < numCams; ++i) {
        std::vector<double> buffs(9);
        for (int j = 0; j < 9; ++j) {
            buffs[j] = std::atof(allLines[curStartLine + i * 9 + j].c_str());
        }
        cv::Mat rotVec(3, 1, CV_64F, buffs.data());
        cv::Mat transVec(3, 1, CV_64F, buffs.data() + 3);
        camPoses[i] = cv::Affine3d(rotVec, transVec);
        std::copy(buffs.begin() + 6, buffs.end(), camIntrinsics[i].begin());
    }

    // 3d points
    curStartLine += 9 * numCams;

    for (int i = 0; i < numPoints; ++i) {
        auto& curPoint3D = point3Ds[i];
        for (int j = 0; j < 3; ++j) {
            curPoint3D[j] = std::atof(allLines[curStartLine + i * 3 + j].c_str());
        }
    }
}
}  // namespace _cv
