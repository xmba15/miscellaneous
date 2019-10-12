function(__fetch_cpp_redis download_module_path download_root)
  set(CPP_REDIS_DOWNLOAD_ROOT ${download_root})
  configure_file(
    ${download_module_path}/cpp_redis-download.cmake
    ${download_root}/CMakeLists.txt
    @ONLY
  )
  unset(CPP_REDIS_DOWNLOAD_ROOT)

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=Release .
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
    ${download_root}/cpp_redis-src
    ${download_root}/cpp_redis-build
  )
endfunction()
