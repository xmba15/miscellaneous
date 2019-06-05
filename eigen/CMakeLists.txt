find_package(Eigen3)

if(TARGET Eigen3::Eigen)
  add_subdirectory(src)
  add_subdirectory(examples)
else()
  message(STATUS "Eigen is not found")
endif(TARGET Eigen3::Eigen)
