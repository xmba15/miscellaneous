list(APPEND EXAMPLES
  "ColorMapping"
  "ColorSpaceConversion"
  "StereoMatching"
  "DisparityVis"
  "Plot"
)

include(example_utility)

add_definitions(-DDATA_PATH="${PROJECT_SOURCE_DIR}/data/images/")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

list(APPEND PUBLIC_LIBS
  ${PROJECT_TARGET_LIB_NAME}
)

list(APPEND PRIVATE_LIBS
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
  target_include_directories(${EXAMPLE}
    PUBLIC
      ${OpenCV_INCLUDE_DIRS}
      ${PCL_INCLUDE_DIRS}
  )
endforeach(EXAMPLE)
