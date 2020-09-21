cmake_minimum_required(VERSION 3.10)

project(draw_coord)
set (CMAKE_CXX_STANDARD 17)

find_package(Threads)

add_subdirectory(third_parties/Pangolin)
find_package(Pangolin REQUIRED)

add_executable(${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/draw_coord_app.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${Pangolin_LIBRARIES}
    Threads::Threads
)
