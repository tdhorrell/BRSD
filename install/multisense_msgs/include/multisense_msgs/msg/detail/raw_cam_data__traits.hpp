// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multisense_msgs:msg/RawCamData.idl
// generated code does not contain a copyright notice

#ifndef MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_DATA__TRAITS_HPP_
#define MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_DATA__TRAITS_HPP_

#include "multisense_msgs/msg/detail/raw_cam_data__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'time_stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multisense_msgs::msg::RawCamData>()
{
  return "multisense_msgs::msg::RawCamData";
}

template<>
inline const char * name<multisense_msgs::msg::RawCamData>()
{
  return "multisense_msgs/msg/RawCamData";
}

template<>
struct has_fixed_size<multisense_msgs::msg::RawCamData>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multisense_msgs::msg::RawCamData>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multisense_msgs::msg::RawCamData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_DATA__TRAITS_HPP_
