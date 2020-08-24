#include <algorithm>
#include <cstdint>
#include <iostream>
#include <list>
#include <string>

class HashTable
{
 public:
    explicit HashTable(const std::int64_t bucket, const int seed = 2020)
        : m_bucket(bucket)
        , m_seed(seed)
    {
        m_table = new std::list<std::string>[bucket];
    }

    virtual ~HashTable()
    {
        delete[] m_table;
    }

    std::int64_t hash(const std::string& key)
    {
        std::int64_t hash = 0;
        for (const char c : key) {
            hash *= m_seed;
            hash += static_cast<int>(c);
        }

        return hash % m_bucket;
    }

    void insertItem(const std::string& key)
    {
        int idx = this->hash(key);

        if (std::find(m_table[idx].begin(), m_table[idx].end(), key) == m_table[idx].end()) {
            m_table[idx].push_back(key);
        }
    }

    void deleteItem(const std::string& key)
    {
        int idx = this->hash(key);
        for (auto it = m_table[idx].begin(); it != m_table[idx].end(); ++it) {
            if (*it == key) {
                m_table[idx].erase(it);
                break;
            }
        }
    }

    void display() const
    {
        std::cout << "------------------------------------------------------\n";
        for (auto i = 0; i < m_bucket; ++i) {
            std::cout << i << ": ";
            for (const auto& key : m_table[i]) {
                std::cout << key << " ";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------------------------------\n";
    }

 private:
    std::int64_t m_bucket;
    int m_seed;
    std::list<std::string>* m_table;
};

int main(int argc, char* argv[])
{
    HashTable h(7);
    h.insertItem("hi");
    h.insertItem("abc");
    h.insertItem("aa");
    h.insertItem("qs");
    h.insertItem("pl");

    h.display();
    h.deleteItem("qs");
    h.display();

    return 0;
}
