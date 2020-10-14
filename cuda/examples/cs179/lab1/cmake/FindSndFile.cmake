include(FindPackageHandleStandardArgs)
unset(SNDFILE_FOUND)

find_path(SNDFILE_INCLUDE_DIR
  NAMES
    sndfile.h
  PATHS
    /usr/include
    /usr/local/include
)

find_library(SNDFILE_LIBRARY
  NAMES
    sndfile
  PATHS
    /usr/lib
    /usr/local/lib
)

set(SNDFILE_INCLUDE_DIRS
  ${SNDFILE_INCLUDE_DIR}
)

set(SNDFILE_LIBRARIES
  ${SNDFILE_LIBRARY}
)

if (SNDFILE_INCLUDE_DIRS AND SNDFILE_LIBRARIES)
  set(SNDFILE_FOUND TRUE)
endif (SNDFILE_INCLUDE_DIRS AND SNDFILE_LIBRARIES)

if (SNDFILE_FOUND)
  if (NOT SndFile_FIND_QUIETLY)
    message(STATUS "Found libsndfile: ${SNDFILE_LIBRARIES}")
  endif (NOT SndFile_FIND_QUIETLY)
else (SNDFILE_FOUND)
  if (SndFile_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find libsndfile")
  endif (SndFile_FIND_REQUIRED)
endif (SNDFILE_FOUND)

mark_as_advanced(SNDFILE_INCLUDE_DIRS SNDFILE_LIBRARIES)
