#include <iostream>
#include <string>

std::string urlify(const std::string& str)
{
    std::string output = "";
    for (auto it = str.cbegin(); it != str.cend(); ++it) {
        if (*it != ' ') {
            output += *it;
        } else {
            if (it + 1 == str.cend()) {
                break;
            }
            if (*(it + 1) == ' ') {
                it++;
            } else {
                output += "%20";
            }
        }
    }

    return output;
}

int main(int argc, char* argv[])
{
    std::string testStr = "Mr John Smith   ";
    std::cout << urlify(testStr) << "\n";
    return 0;
}
