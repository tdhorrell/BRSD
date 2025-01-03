// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multisense_msgs:msg/RawImuData.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multisense_msgs/msg/detail/raw_imu_data__rosidl_typesupport_introspection_c.h"
#include "multisense_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multisense_msgs/msg/detail/raw_imu_data__functions.h"
#include "multisense_msgs/msg/detail/raw_imu_data__struct.h"


// Include directives for member types
// Member `time_stamp`
#include "builtin_interfaces/msg/time.h"
// Member `time_stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void RawImuData__rosidl_typesupport_introspection_c__RawImuData_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multisense_msgs__msg__RawImuData__init(message_memory);
}

void RawImuData__rosidl_typesupport_introspection_c__RawImuData_fini_function(void * message_memory)
{
  multisense_msgs__msg__RawImuData__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_member_array[4] = {
  {
    "time_stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawImuData, time_stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawImuData, x),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawImuData, y),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "z",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__RawImuData, z),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_members = {
  "multisense_msgs__msg",  // message namespace
  "RawImuData",  // message name
  4,  // number of fields
  sizeof(multisense_msgs__msg__RawImuData),
  RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_member_array,  // message members
  RawImuData__rosidl_typesupport_introspection_c__RawImuData_init_function,  // function to initialize message memory (memory has to be allocated)
  RawImuData__rosidl_typesupport_introspection_c__RawImuData_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_type_support_handle = {
  0,
  &RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multisense_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multisense_msgs, msg, RawImuData)() {
  RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_type_support_handle.typesupport_identifier) {
    RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &RawImuData__rosidl_typesupport_introspection_c__RawImuData_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
