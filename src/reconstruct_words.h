/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "arith_utils.h"
#include "rlwe_client.h"
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lwe_ann {

template <typename word, typename plain_word>
inline std::vector<word>
reconstruct_from_plain_words_lwe(const std::vector<plain_word> &chunked,
                                 size_t chunk_size) {
  if (chunked.size() % chunk_size != 0) {
    throw std::invalid_argument(
        "chunked.size() must be divisible by chunk_size");
  }
  if (sizeof(word) < sizeof(plain_word) * chunk_size) {
    throw std::invalid_argument(
        "sizeof(word) must be greater than sizeof(plain_word) * chunk_size");
  }

  std::vector<word> res(chunked.size() / chunk_size);

  word mask = ((word)1) << (sizeof(plain_word) * 8 * chunk_size - 1);
  for (size_t i = 0; i < res.size(); i++) {
    word val = 0;
    for (int j = static_cast<int>(chunk_size) - 1; j >= 0; j--) {
      val <<= sizeof(plain_word) * 8;
      val |= zero_extend<word>(chunked[i * chunk_size + j]);
    }
    if constexpr (std::is_signed_v<word>) {
      // sign-extend
      res[i] = (val ^ mask) - mask;
    } else {
      res[i] = val;
    }
  }
  return res;
}

template <typename word, typename plain_word>
inline RLWECiphertextPack<word> reconstruct_from_plain_words_rlwe(
    const std::vector<plain_word> &chunked, size_t chunk_size, size_t N,
    double scale, size_t round_shift_bits) {
  auto tmp =
      reconstruct_from_plain_words_lwe<word, plain_word>(chunked, chunk_size);
  RLWECiphertextPack<word> ctxt;
  auto tmp_mid = tmp.begin() + tmp.size() / 2;
  ctxt.a_data_ = PolynomialPack<word>(tmp.begin(), tmp_mid);
  ctxt.b_data_ = PolynomialPack<word>(tmp_mid, tmp.end());
  ctxt.a_data_.is_ntt_ = false;
  ctxt.a_data_.batch_size_ = tmp.size() / 2 / N;
  ctxt.b_data_.is_ntt_ = false;
  ctxt.b_data_.batch_size_ = tmp.size() / 2 / N;
  ctxt.scale_ = scale;
  ctxt.round_shift_bits_ = round_shift_bits;
  return ctxt;
}

template <typename T>
inline std::vector<T> reorganize_2d(const std::vector<T> &decrypted_res,
                                    size_t N, size_t num_gpus, size_t dim_last,
                                    size_t padded_dim_last) {
  if (padded_dim_last % num_gpus != 0) {
    throw std::invalid_argument(
        "padded_dim_last must be divisible by num_gpus");
  }
  size_t padded_dim_last_per_gpu = padded_dim_last / num_gpus;
  size_t num_poly_per_gpu = div_ceil<size_t>(padded_dim_last_per_gpu, N);
  if (decrypted_res.size() != num_poly_per_gpu * N * num_gpus) {
    throw std::invalid_argument("decrypted_res.size() must be equal to "
                                "num_poly_per_gpu * N * num_gpus");
  }

  std::vector<T> res(dim_last);

  size_t remaining_dim_last = dim_last;
  for (size_t i = 0; i < num_gpus; i++) {
    size_t copy_size =
        std::min<size_t>(remaining_dim_last, padded_dim_last_per_gpu);
    if (copy_size == 0)
      break;

    auto decrypted_res_begin = decrypted_res.begin() + i * num_poly_per_gpu * N;
    auto decrypted_res_end = decrypted_res_begin + copy_size;
    auto res_begin = res.begin() + i * padded_dim_last_per_gpu;

    std::copy(decrypted_res_begin, decrypted_res_end, res_begin);
    remaining_dim_last -= copy_size;
  }
  return res;
}

template <typename plain_word>
inline std::vector<plain_word>
reorganize_2d_without_packing(const std::vector<plain_word> &decrypted_res,
                              size_t dim_last, size_t padded_dim_last,
                              size_t batch_size) {
  if (decrypted_res.size() != padded_dim_last * batch_size) {
    throw std::invalid_argument(
        "decrypted_res.size() must be equal to padded_dim_last * batch_size");
  }

  std::vector<plain_word> res(dim_last * batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    std::copy(decrypted_res.begin() + i * padded_dim_last,
              decrypted_res.begin() + i * padded_dim_last + dim_last,
              res.begin() + i * dim_last);
  }
  return res;
}

template <typename T>
inline std::vector<T> reorganize_3d(const std::vector<T> &decrypted_res,
                                    size_t N, size_t dim_last,
                                    size_t padded_dim_last) {
  size_t num_poly = div_ceil<size_t>(padded_dim_last, N);
  if (decrypted_res.size() != num_poly * N) {
    throw std::invalid_argument("decrypted_res.size() must be equal to "
                                "num_poly * N");
  }

  return std::vector<T>(decrypted_res.begin(),
                        decrypted_res.begin() + dim_last);
}

template <typename word, typename plain_word, typename T = plain_word>
inline std::vector<T> decrypt_pir_res_rlwe_2d(
    RLWEClient<word> &rlwe_client, const RLWESecret<word> &rlwe_secret,
    const std::vector<plain_word> &pir_res, size_t N, size_t num_gpus,
    size_t dim_last, size_t padded_dim_last, size_t num_chunks, double scale,
    size_t round_shift_bits) {
  auto ctxt = reconstruct_from_plain_words_rlwe<word, plain_word>(
      pir_res, num_chunks, N, scale, round_shift_bits);
  PolynomialPack<T> decrypted_res;
  rlwe_client.decrypt(decrypted_res, ctxt, rlwe_secret);
  return reorganize_2d(decrypted_res, N, num_gpus, dim_last, padded_dim_last);
}

template <typename word, typename plain_word, typename T = plain_word>
inline std::vector<T> decrypt_pir_res_rlwe_3d(
    RLWEClient<word> &rlwe_client, const RLWESecret<word> &rlwe_secret,
    const std::vector<plain_word> &pir_res, size_t N, size_t dim_last,
    size_t padded_dim_last, size_t num_chunks, double scale,
    size_t round_shift_bits) {
  // First decryption
  auto ctxt1 = reconstruct_from_plain_words_rlwe<word, plain_word>(
      pir_res, num_chunks, N, scale, round_shift_bits);
  PolynomialPack<plain_word> decrypted_res;
  rlwe_client.decrypt(decrypted_res, ctxt1, rlwe_secret);

  // Second decryption
  auto ctxt2 = reconstruct_from_plain_words_rlwe<word, plain_word>(
      decrypted_res, num_chunks, N, scale, round_shift_bits);
  PolynomialPack<T> decrypted_res_2;
  rlwe_client.decrypt(decrypted_res_2, ctxt2, rlwe_secret);

  return reorganize_3d<T>(decrypted_res_2, N, dim_last, padded_dim_last);
}

} // namespace lwe_ann
