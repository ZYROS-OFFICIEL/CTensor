#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MyProject::myproject" for configuration ""
set_property(TARGET MyProject::myproject APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(MyProject::myproject PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmyproject.a"
  )

list(APPEND _cmake_import_check_targets MyProject::myproject )
list(APPEND _cmake_import_check_files_for_MyProject::myproject "${_IMPORT_PREFIX}/lib/libmyproject.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
