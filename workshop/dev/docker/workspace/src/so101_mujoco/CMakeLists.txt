cmake_minimum_required(VERSION 3.8)
project(so101_mujoco)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  mujoco
  meshes
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/so101_mujoco_bridge.py
  scripts/so101_mujoco_viewer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
