find_package(Boost REQUIRED COMPONENTS
  serialization
  container
)

if (${Boost_FOUND})
message(STATUS "Boost found")

add_executable(test_serialization
  ${CMAKE_CURRENT_LIST_DIR}/TestSerialization.cpp
)
target_link_libraries(test_serialization
  Boost::serialization
)

add_executable(test_container
  ${CMAKE_CURRENT_LIST_DIR}/TestContainer.cpp
)
target_link_libraries(test_container
  Boost::container
)

endif()
