cmake_minimum_required(VERSION 3.8)
project(long_term_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++17 )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_system -DGTSAM_USE_SYSTEM_EIGEN=ON")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nano_gicp QUIET)
find_package(fast_lio REQUIRED)

find_package(GTSAM REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS system timer thread serialization chrono)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

# ---------------- KISS-Matcher ----------------
# colcon 빌드 환경에서 /usr/local 탐색 경로 추가
list(APPEND CMAKE_PREFIX_PATH /usr/local)
find_package(robin QUIET)        # kiss_matcherConfig.cmake에 robin 의존성이 누락되어 있어 먼저 로드
find_package(kiss_matcher QUIET)

# TARGET 존재 여부로 판단 (find_package 결과 변수보다 신뢰성 높음)
if(NOT TARGET kiss_matcher::kiss_matcher_core)
  message(STATUS "[long_term_mapping] kiss_matcher target not found, building from bundled source")
  set(USE_SYSTEM_TBB    ON  CACHE BOOL "Use system TBB"    FORCE)
  set(USE_SYSTEM_EIGEN3 ON  CACHE BOOL "Use system Eigen3" FORCE)
  set(USE_SYSTEM_ROBIN  OFF CACHE BOOL "Use system ROBIN"  FORCE)
  add_subdirectory(include/kiss_matcher)
else()
  message(STATUS "[long_term_mapping] Found kiss_matcher::kiss_matcher_core target")
endif()
# ----------------------------------------------

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  include)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  nano_gicp
  pcl_ros
  pcl_conversions  
)

ament_export_dependencies(rosidl_default_runtime)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()


add_executable(LTmapping src/long_term_mapping.cpp
                  include/solid/solid.cpp)

target_link_libraries(LTmapping
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
  kiss_matcher::kiss_matcher_core
)

target_include_directories(LTmapping PRIVATE ${PYTHON_INCLUDE_DIRS})

ament_target_dependencies(LTmapping
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  nano_gicp
  fast_lio
)


# ---------------- Install --------------- #
install(TARGETS LTmapping
  DESTINATION lib/${PROJECT_NAME})
  
install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
