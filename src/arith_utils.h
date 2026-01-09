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

#include "type_utils.h"
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lwe_ann {

template <typename word> constexpr size_t log2_ceil(word x) {
  x -= 1;
  size_t log2 = 0;
  while (x > 0) {
    x >>= 1;
    log2++;
  }
  return log2;
}

template <typename word> constexpr size_t log2_floor(word x) {
  size_t log2 = 0;
  while (x > 1) {
    x >>= 1;
    log2++;
  }
  return log2;
}

template <typename word> constexpr bool is_pow2(word x) {
  return (x & (x - 1)) == 0;
}

template <typename word> constexpr word div_ceil(word x, word y) {
  return (x + y - 1) / y;
}

template <typename word> constexpr word pad_by(word size, word granularity) {
  return div_ceil<word>(size, granularity) * granularity;
}

template <typename small_word, typename big_word>
small_word bit_truncate(big_word x) {
  static_assert(sizeof(big_word) >= sizeof(small_word),
                "big_word cannot be smaller than small_word");
  // Unsigned to unsigned conversion always results in a correct truncation
  return *reinterpret_cast<small_word *>(&x);
}

template <typename big_word, typename small_word>
big_word zero_extend(small_word x) {
  static_assert(sizeof(big_word) >= sizeof(small_word),
                "big_word cannot be smaller than small_word");
  using unsigned_small_word = make_unsigned_t<small_word>;
  using unsigned_big_word = make_unsigned_t<big_word>;
  return static_cast<unsigned_big_word>(
      *reinterpret_cast<unsigned_small_word *>(&x));
}

template <typename big_word, typename small_word>
big_word sign_extend(small_word x) {
  static_assert(sizeof(big_word) >= sizeof(small_word),
                "big_word cannot be smaller than small_word");
  using signed_small_word = make_signed_t<small_word>;
  using signed_big_word = make_signed_t<big_word>;
  return static_cast<signed_big_word>(
      *reinterpret_cast<signed_small_word *>(&x));
}

/**
 * Extended Euclidean Algorithm finds x, y such that ax + by = gcd(a, b)
 *
 * @tparam word the type of the numbers (signed)
 * @param a first number
 * @param b seconds number
 * @param x x = a^-1 * gcd(a, b) mod b
 * @param y y = b^-1 * gcd(a, b) mod a
 * @return gcd(a, b)
 */
template <typename word> word extended_gcd(word a, word b, word &x, word &y) {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");
  // Base Case
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }

  word x1, y1;
  word gcd = extended_gcd(b % a, a, x1, y1);

  // Update x and y using results of
  // recursive call
  x = y1 - (b / a) * x1;
  y = x1;
  return gcd;
}

template <typename uword> struct barrett_util {
  static_assert(is_valid_unsigned_word_v<uword>,
                "Barrett util only supports unsigned words");
  using dword = make_dword_t<uword>;
  using sword = make_signed_t<uword>;
  const uword modulus_;
  const uword half_modulus_;

public:
  const uword barrett_reciprocal_single_;
  const size_t barrett_k_double_;
  const uword barrett_reciprocal_double_;

  barrett_util(uword modulus)
      : modulus_(modulus), half_modulus_(modulus >> 1),
        barrett_reciprocal_single_(std::numeric_limits<uword>::max() / modulus),
        barrett_k_double_(sizeof(uword) * 8 - 1 + log2_ceil(modulus)),
        barrett_reciprocal_double_(static_cast<uword>(
            (((dword)1) << sizeof(uword) * 8 - 1 + log2_ceil(modulus)) /
            modulus)) {
    if (is_pow2(modulus)) {
      throw std::invalid_argument("Modulus cannot be a power of 2");
    }
    if (modulus <= 2) {
      throw std::invalid_argument("Modulus is too small");
    }
    if (modulus >= (((uword)1) << (sizeof(uword) * 8 - 1))) {
      throw std::invalid_argument("Modulus is too large");
    }
  }

  inline uword add(uword x, uword y) const {
    uword res = x + y;
    res -= (res >= modulus_) * modulus_;
    return res;
  }

  inline uword sub(uword x, uword y) const {
    using sword = make_signed_t<uword>;
    sword res = static_cast<sword>(x) - static_cast<sword>(y);
    res += (res < 0) * static_cast<sword>(modulus_);
    return static_cast<uword>(res);
  }

  inline uword neg(uword x) const { return (x != 0) * modulus_ - x; }

  inline uword reduce_single(uword x) const {
    dword tmp = static_cast<dword>(x) * barrett_reciprocal_single_;
    x -= static_cast<uword>(tmp >> (sizeof(uword) * 8)) * modulus_;
    x -= (x >= modulus_) * modulus_;
    return x;
  }

  inline uword reduce_double(dword x) const {
    constexpr size_t uword_bits = sizeof(uword) * 8;
    uword x_low = static_cast<uword>(x);
    uword x_high = static_cast<uword>(x >> uword_bits);
    dword tmp_low = static_cast<dword>(x_low) * barrett_reciprocal_double_;
    dword tmp_high = static_cast<dword>(x_high) * barrett_reciprocal_double_;
    dword tmp_shift = (tmp_low >> barrett_k_double_) +
                      (tmp_high >> (barrett_k_double_ - uword_bits));
    // tmp = tmp_low + tmp_high * 2^32 (if uword is 32 bits)
    // tmp_shift = tmp << barrett_k_double_
    uword res = static_cast<uword>(x - tmp_shift * modulus_);
    res -= (res >= modulus_) * modulus_;
    return res;
  }

  inline uword mul(uword x, uword y) const {
    dword tmp = static_cast<dword>(x) * y;
    return reduce_double(tmp);
  }

  inline sword convert_to_signed_range(uword x) const {
    return static_cast<sword>(x) -
           (x > half_modulus_) * static_cast<sword>(modulus_);
  }

  inline uword convert_to_unsigned_range(sword x) const {
    x += (x < 0) * static_cast<sword>(modulus_);
    return static_cast<uword>(x);
  }
};

} // namespace lwe_ann
