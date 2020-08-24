#include <iostream>
#include <string>

bool isOneAway(const std::string& fstr, const std::string& sstr)
{
    int s1 = fstr.size();
    int s2 = sstr.size();

    if (std::abs(s1 - s2) > 1) {
        return false;
    }

    if (s1 == s2) {
        if (fstr == sstr) {
            return false;
        }

        int count = 0;
        for (int i = 0; i < s1; ++i) {
            if (fstr[i] != sstr[i]) {
                count++;
                if (count > 1) {
                    return false;
                }
            }
        }
        return true;
    }

    int i = 0, j = 0;

    while (i < s1 && j << s2) {
        if (fstr[i] == sstr[j]) {
            i++;
            j++;
        } else {
            if (i + 1 == s1 || j + 1 == s2) {
                return false;
            }
            if (fstr.substr(i + 1) == sstr.substr(i) || fstr.substr(i) == sstr.substr(i + 1)) {
                return true;
            }
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    std::cout << isOneAway("pale", "ple") << "\n";
    std::cout << isOneAway("pales", "pale") << "\n";
    std::cout << isOneAway("pales", "bale") << "\n";
    std::cout << isOneAway("pale", "bake") << "\n";
    return 0;
}
