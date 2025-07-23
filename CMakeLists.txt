cmake_minimum_required(VERSION 3.20.0)
project(SimpleProject)

find_package(LLVM REQUIRED CONFIG)
set(llvm_all "")
set(non_llvm_libs "")
foreach(llvm_lib ${LLVM_AVAILABLE_LIBS})
  # remove "LLVM" library:
  message(STATUS "Processing LLVM library: ${llvm_lib}")
  if (llvm_lib STREQUAL "LLVM")
    message(STATUS "Skipping LLVM library itself")
    continue()
  endif()
  # skip if a shared library
  get_target_property(target_type ${llvm_lib} TYPE)
  if (target_type STREQUAL "SHARED_LIBRARY")
    message(STATUS "Skipping shared library: ${llvm_lib}")
    continue()
  endif()
  get_target_property(target_interface_link_libraries ${llvm_lib} INTERFACE_LINK_LIBRARIES)
  foreach(link_lib ${target_interface_link_libraries})
    # if its not an LLVM library, skip it
    if (NOT link_lib MATCHES "^LLVM")
      message(STATUS "adding non-LLVM library: ${link_lib}")
      list(APPEND non_llvm_libs ${link_lib})
    endif()
  endforeach()
  
  list(APPEND llvm_all ${llvm_lib})
endforeach()

message(STATUS "LLVM libraries: ${llvm_all}")
message(STATUS "Non-LLVM libraries: ${non_llvm_libs}")
llvm_expand_dependencies(llvm_expanded ${llvm_all})
message(STATUS "Expanded LLVM libraries: ${llvm_expanded}")

#add_executable(SimpleProject dummy.cpp)
#target_link_libraries(SimpleProject ${llvm_all})
#get_target_property(SimpleProjectLibs SimpleProject INTERFACE_LINK_LIBRARIES)

message(STATUS "_START_${llvm_expanded}_END_")
