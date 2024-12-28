// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multisense_msgs:msg/RawLidarData.idl
// generated code does not contain a copyright notice

#ifndef MULTISENSE_MSGS__MSG__DETAIL__RAW_LIDAR_DATA__TRAITS_HPP_
#define MULTISENSE_MSGS__MSG__DETAIL__RAW_LIDAR_DATA__TRAITS_HPP_

#include "multisense_msgs/msg/detail/raw_lidar_data__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'time_start'
// Member 'time_end'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multisense_msgs::msg::RawLidarData>()
{
  return "multisense_msgs::msg::RawLidarData";
}

template<>
inline const char * name<multisense_msgs::msg::RawLidarData>()
{
  return "multisense_msgs/msg/RawLidarData";
}

template<>
struct has_fixed_size<multisense_msgs::msg::RawLidarData>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multisense_msgs::msg::RawLidarData>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multisense_msgs::msg::RawLidarData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTISENSE_MSGS__MSG__DETAIL__RAW_LIDAR_DATA__TRAITS_HPP_
