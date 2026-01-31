cmake_minimum_required(VERSION 3.8)
project(huaray_camera)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
# enforcing cleaner code.
add_definitions(-Wall -Werror)

find_library(
  CAMERA_LIB
  NAMES MVSDK
  PATHS ./huaraySDK/lib
  NO_DEFAULT_PATH
  REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED src/huaray_camera_node.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include
                                                  huaraySDK/include
                                                  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CAMERA_LIB} rt)
message(STATUS ${CAMERA_LIB})

rclcpp_components_register_node(
  ${PROJECT_NAME} PLUGIN HuarayCamera::HuarayCameraNode EXECUTABLE
  ${PROJECT_NAME}_node)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_copyright ament_cmake_cpplint
       ament_cmake_uncrustify)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)

# cmake_minimum_required(VERSION 3.22) project(huaray_camera)

# set(CMAKE_CXX_STANDARD 11) find_library( CAMERA_LIB NAMES MVSDK HINTS
# ./camera_lib/x86 REQUIRED) find_package( OpenCV REQUIRED )

# include_directories(camera_include)

# include_directories( ${OpenCV_INCLUDE_DIRS} )

# add_executable(huaray_camera main.cpp Camera.cpp Camera.h auto_aim.cpp
# armor.cpp auto_aim.hpp armor.hpp ) message(STATUS ${CAMERA_LIB})

# message(${CAMERA_LIB}) target_link_libraries(huaray_camera  ${CAMERA_LIB}
# ${OpenCV_LIBS} rt)
