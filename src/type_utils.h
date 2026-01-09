/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace lwe_ann {

using std::is_signed;
using std::is_signed_v;
using std::is_unsigned;
using std::is_unsigned_v;
using std::make_signed;
using std::make_signed_t;
using std::make_unsigned;
using std::make_unsigned_t;

template <typename... T> constexpr bool always_false = false;

template <typename T> struct is_valid_signed_word {
  static constexpr bool value =
      std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;
};
template <typename T>
constexpr bool is_valid_signed_word_v = is_valid_signed_word<T>::value;

template <typename T> struct is_valid_unsigned_word {
  static constexpr bool value =
      std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
      std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;
};
template <typename T>
constexpr bool is_valid_unsigned_word_v = is_valid_unsigned_word<T>::value;

// A valid word is an integral type that is not const, volatile, and is not
// bool.
template <typename T> struct is_valid_word {
  static constexpr bool value =
      is_valid_signed_word_v<T> || is_valid_unsigned_word_v<T>;
};
template <typename T> constexpr bool is_valid_word_v = is_valid_word<T>::value;

namespace detail {

// Compilation will fail if the type is not a valid word.
template <typename T> struct __make_dword_impl;

template <> struct __make_dword_impl<int8_t> {
  using type = int16_t;
};

template <> struct __make_dword_impl<int16_t> {
  using type = int32_t;
};

template <> struct __make_dword_impl<int32_t> {
  using type = int64_t;
};

template <> struct __make_dword_impl<int64_t> {
  using type = __int128_t;
};

template <> struct __make_dword_impl<uint8_t> {
  using type = uint16_t;
};

template <> struct __make_dword_impl<uint16_t> {
  using type = uint32_t;
};

template <> struct __make_dword_impl<uint32_t> {
  using type = uint64_t;
};

template <> struct __make_dword_impl<uint64_t> {
  using type = __uint128_t;
};

} // namespace detail

template <typename T> struct make_dword {
  using type = typename detail::__make_dword_impl<T>::type;
};
template <typename T> using make_dword_t = typename make_dword<T>::type;

template <typename T> struct make_signed_dword {
  using type = make_signed_t<make_dword_t<T>>;
};
template <typename T> using make_signed_dword_t = typename make_signed_dword<T>::type;

template <typename T> struct make_unsigned_dword {
  using type = make_unsigned_t<make_dword_t<T>>;
};
template <typename T> using make_unsigned_dword_t = typename make_unsigned_dword<T>::type;

} // namespace lwe_ann