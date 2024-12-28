# CMake generated Testfile for 
# Source directory: /home/brsd/microros_ws/src/xacro/test
# Build directory: /home/brsd/microros_ws/build/xacro/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pytest "/usr/bin/python3" "-u" "/opt/ros/foxy/share/ament_cmake_test/cmake/run_test.py" "/home/brsd/microros_ws/build/xacro/test_results/xacro/pytest.xunit.xml" "--package-name" "xacro" "--output-file" "/home/brsd/microros_ws/build/xacro/ament_cmake_pytest/pytest.txt" "--env" "AMENT_PREFIX_PATH=/home/brsd/microros_ws/build/xacro/test/test_ament_index:/home/brsd/microros_ws/install/micro_ros_setup:/home/brsd/microros_ws/install/micro_ros_agent:/home/brsd/microros_ws/install/micro_ros_msgs:/home/brsd/microros_ws/install/jetson_comm:/home/brsd/microros_ws/install/drive_base_msgs:/opt/ros/foxy" "--command" "/usr/bin/python3" "-u" "-m" "pytest" "/home/brsd/microros_ws/src/xacro/test/." "-o" "cache_dir=/home/brsd/microros_ws/build/xacro/test/ament_cmake_pytest/pytest/.cache" "--junit-xml=/home/brsd/microros_ws/build/xacro/test_results/xacro/pytest.xunit.xml" "--junit-prefix=xacro")
set_tests_properties(pytest PROPERTIES  LABELS "pytest" TIMEOUT "60" WORKING_DIRECTORY "/home/brsd/microros_ws/src/xacro/test" _BACKTRACE_TRIPLES "/opt/ros/foxy/share/ament_cmake_test/cmake/ament_add_test.cmake;118;add_test;/opt/ros/foxy/share/ament_cmake_pytest/cmake/ament_add_pytest_test.cmake;165;ament_add_test;/home/brsd/microros_ws/src/xacro/test/CMakeLists.txt;10;ament_add_pytest_test;/home/brsd/microros_ws/src/xacro/test/CMakeLists.txt;0;")
add_test(xacro_cmake "/home/brsd/microros_ws/src/xacro/test/test-cmake.sh" "/home/brsd/microros_ws/src/xacro/test/test-xacro-cmake")
set_tests_properties(xacro_cmake PROPERTIES  _BACKTRACE_TRIPLES "/home/brsd/microros_ws/src/xacro/test/CMakeLists.txt;15;add_test;/home/brsd/microros_ws/src/xacro/test/CMakeLists.txt;0;")
