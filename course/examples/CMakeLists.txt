list(APPEND EXAMPLES
  "ColorMapping"
)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)

include(example_utility)
list(APPEND PUBLIC_LIBS
  ${OPENCV_LIBS}
  Eigen3::Eigen
)

list(APPEND PRIVATE_LIBS
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
   )
endforeach(EXAMPLE)
