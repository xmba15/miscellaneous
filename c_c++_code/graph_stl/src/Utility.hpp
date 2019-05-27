/**
 * @file    Utility.hpp
 *
 * @brief   Utility Functions
 *
 * @author  btran
 *
 * @date    2019-05-21
 *
 * Copyright (c) organization
 *
 */

#include <memory>

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
std::ostream &operator<<(std::ostream &os, const Container<T> &container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin();
         it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}
