find_package(Open3D)

if(${Open3D_FOUND})
message(STATUS "Open3d found")

add_executable(test_load_open3d
  ${CMAKE_CURRENT_LIST_DIR}/TestOpen3D.cpp
)
target_link_libraries(test_load_open3d
  ${Open3D_LIBRARIES}
)

target_include_directories(test_load_open3d
  PUBLIC ${Open3D_INCLUDE_DIRS}
)

endif()
