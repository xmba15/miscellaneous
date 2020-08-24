#include <iostream>
#include <string>

bool allUnique(const std::string& str)
{
    int count['z' - 'a'];
    std::fill_n(count, 'z' - 'a', 0);
    for (const char c : str) {
        count[std::tolower(c) - 'a']++;
    }

    for (int i = 0; i < 'z' - 'a'; ++i) {
        if (count[i] > 1) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    std::cout << allUnique("adele") << "\n";
    std::cout << allUnique("home") << "\n";

    return 0;
}
