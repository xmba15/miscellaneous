/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#if defined(__GNUC__) && __GNUC__ >= 8
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#if defined(__GNUC__) && __GNUC__ >= 8
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

inline std::vector<std::string> parseDirectory(const std::string& path, const bool sorted = true)
{
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(path)) {
        files.emplace_back(entry.path());
    }

    if (sorted) {
        std::sort(files.begin(), files.end());
    }
    return files;
}
