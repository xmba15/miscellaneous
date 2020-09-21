cmake_minimum_required(VERSION 3.10)

project(julia_set)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

if(${OpenGL_FOUND} AND ${GLUT_FOUND})
  file(GLOB BASIC_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  )

  cuda_add_executable(${PROJECT_NAME}
    ${BASIC_SOURCE_FILES}
  )

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      OpenGL::OpenGL
      ${GLUT_LIBRARY}
  )

  cuda_include_directories(${PROJECT_NAME}
    ${GLUT_INCLUDE_DIR}
  )
else(cond)
  message(STATUS "Opengl NOT FOUND")
endif(${OpenGL_FOUND} AND ${GLUT_FOUND})
