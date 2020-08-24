#include <iostream>
#include <string>

std::string compressString(const std::string& str)
{
    int curCount = 0;
    char curChar;
    std::string result = "";

    for (auto it = str.cbegin(); it != str.cend(); ++it) {
        if (it == str.cbegin()) {
            curCount = 1;
            curChar = std::tolower(*it);
        } else if (*it != curChar) {
            result += curChar;
            result += std::to_string(curCount);
            curChar = std::tolower(*it);
            curCount = 1;
        } else {
            curCount++;
        }

        if ((it + 1) == str.end()) {
            result += curChar;
            result += std::to_string(curCount);
        }
    }

    return result;
}

int main(int argc, char* argv[])
{
    std::cout << compressString("aabcccccaaa") << "\n";
    return 0;
}
