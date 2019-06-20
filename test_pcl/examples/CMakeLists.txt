find_package(PCL)

if(${PCL_FOUND})
message(STATUS "PCL found")

add_executable(test_octree
  ${CMAKE_CURRENT_LIST_DIR}/TestOctree.cpp
)

target_link_libraries(test_octree
  ${PCL_LIBRARIES}
)

target_include_directories(test_load_open3d
  PUBLIC ${PCL_INCLUDE_DIRS}
)

endif(${PCL_FOUND})
