// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multisense_msgs:msg/RawLidarCal.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multisense_msgs/msg/detail/raw_lidar_cal__rosidl_typesupport_introspection_c.h"
#include "multisense_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multisense_msgs/msg/detail/raw_lidar_cal__functions.h"
#include "multisense_msgs/msg/detail/raw_lidar_cal__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multisense_msgs__msg__RawLidarCal__init(message_memory);
}

void RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_fini_function(void * message_memory)
{
  multisense_msgs__msg__RawLidarCal__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_member_array[2] = {
  {
    "laser_to_spindle",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    16,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawLidarCal, laser_to_spindle),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "camera_to_spindle_fixed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    16,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawLidarCal, camera_to_spindle_fixed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_members = {
  "multisense_msgs__msg",  // message namespace
  "RawLidarCal",  // message name
  2,  // number of fields
  sizeof(multisense_msgs__msg__RawLidarCal),
  RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_member_array,  // message members
  RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_init_function,  // function to initialize message memory (memory has to be allocated)
  RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_type_support_handle = {
  0,
  &RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multisense_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multisense_msgs, msg, RawLidarCal)() {
  if (!RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_type_support_handle.typesupport_identifier) {
    RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &RawLidarCal__rosidl_typesupport_introspection_c__RawLidarCal_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
