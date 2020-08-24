#include <iostream>
#include <string>

bool isPalindromePermutation(const std::string& str)
{
    int count['z' - 'a'];
    std::fill_n(count, 'z' - 'a', 0);
    for (const char c : str) {
        count[std::tolower(c) - 'a']++;
    }

    int countOdd = 0;
    for (int i = 0; i < 'z' - 'a'; ++i) {
        if (count[i] % 2 == 1) {
            countOdd++;
            if (countOdd > 1) {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    std::string input = "Tact Coa";
    std::string input2 = "Tact Coattt";
    std::cout << isPalindromePermutation(input) << "\n";
    std::cout << isPalindromePermutation(input2) << "\n";

    return 0;
}
