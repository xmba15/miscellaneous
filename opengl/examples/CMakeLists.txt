list(APPEND EXAMPLES
  "HelloWorld"
  "SimplePoints"
)

include(example_utility)

add_definitions(-DDATA_PATH="${PROJECT_SOURCE_DIR}/data")

list(APPEND PUBLIC_LIBS
  ${PROJECT_TARGET_LIB_NAME}
)

list(APPEND PRIVATE_LIBS
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARY}
)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
  target_include_directories(${EXAMPLE}
    PUBLIC
      ${OPENGL_INCLUDE_DIRS}
      ${GLUT_INCLUDE_DIRS}
  )
endforeach(EXAMPLE)
