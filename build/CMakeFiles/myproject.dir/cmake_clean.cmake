file(REMOVE_RECURSE
  "libmyproject.a"
  "libmyproject.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/myproject.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
