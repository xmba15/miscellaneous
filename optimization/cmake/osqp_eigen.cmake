function(__fetch_osqp_eigen download_module_path download_root)
  set(OSQP_EIGEN_DOWNLOAD_ROOT ${download_root})
  configure_file(
    ${download_module_path}/osqp_eigen-download.cmake
    ${download_root}/CMakeLists.txt
    @ONLY
  )
  unset(OSQP_EIGEN_DOWNLOAD_ROOT)

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY
      ${download_root}
  )
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY
      ${download_root}
  )

  add_subdirectory(
    ${download_root}/osqp_eigen-src
    ${download_root}/osqp_eigen-build
  )
  message(STATUS "${download_root}/osqp_eigen-src")
endfunction()
