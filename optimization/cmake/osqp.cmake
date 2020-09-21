function(__fetch_osqp download_module_path download_root)
  set(OSQP_DOWNLOAD_ROOT ${download_root})
  configure_file(
    ${download_module_path}/osqp-download.cmake
    ${download_root}/CMakeLists.txt
    @ONLY
  )
  unset(OSQP_DOWNLOAD_ROOT)

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
    ${download_root}/osqp-src
    ${download_root}/osqp-build
  )
  message(STATUS "${download_root}/osqp-src")
endfunction()
