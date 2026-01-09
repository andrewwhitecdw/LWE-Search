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

#include "lwe_server.h"
#include "rlwe_compress.h"
#include "rlwe_server.h"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace lwe_ann {

template <typename l_word, typename r_word, typename plain_word>
class SimplePIRServer {
private:
  void
  db_gemm(std::vector<logical_data_t<cudastf::slice<l_word>>> &multi_gpu_res,
          logical_data_t<cudastf::slice<l_word>> &data, size_t M);

  static constexpr uint32_t tile_m_min_ = 16;
  static constexpr uint32_t tile_m_max_ = 64;
  static constexpr uint32_t tile_n_max_ = 256;
  static constexpr uint32_t tile_k_max_ = 32;

  LWEContextPtr<l_word> lwe_;
  RLWEContextPtr<r_word> rlwe_ = nullptr;
  // This context uses a smaller (lwe_->n_ - 1) degree.
  // This enables us to perform GEMM for the subsequent dimension reduction
  // steps more efficiently.
  LWEContextPtr<l_word> lwe_small_ = nullptr;

  // Num dimension reduction steps x num_gpus
  std::vector<std::vector<RLWECompressPtr<l_word, r_word>>> rlwe_compress_;
  size_t num_rlwe_plain_word_chunks_ = 0;

  const size_t lwe_round_shift_bits_;
  const size_t rlwe_round_shift_bits_;

  const std::vector<size_t> orig_dims_;
  // Currently only up to 3D constructions are supported
  std::vector<size_t> padded_dims_;

  std::vector<seal::prng_seed_type> a_seed_;

  std::vector<logical_data_t<cudastf::slice<l_word>>> second_dim_a_;

  std::vector<int> device_ids_;
  // Used for the first dimension reduction step
  std::vector<plain_word *> device_db_ptrs_;

  const size_t max_batch_size_;
  std::vector<logical_data_t<cudastf::slice<l_word>>> query_ld_;
  std::vector<logical_data_t<cudastf::slice<r_word>>> conv_ctxt_ld_;

  void prepare_first_reduction(std::vector<plain_word> &&db);
  void prepare_first_reduction_2d(std::vector<plain_word> &&db);

  void merge_data_across_gpus(
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
      std::vector<std::vector<logical_data_t<cudastf::slice<plain_word>>>>
          &reinterpreted_res);

  void second_dim_reduction(
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &new_db,
      logical_data_t<cudastf::slice<l_word>> &query);

public:
  SimplePIRServer(std::vector<plain_word> &&db, const std::vector<size_t> &dims,
                  LWEContextPtr<l_word> lwe, size_t lwe_round_shift_bits = 0,
                  RLWEContextPtr<r_word> rlwe = nullptr,
                  size_t rlwe_round_shift_bits = 0,
                  const std::vector<int> &device_ids = {0},
                  size_t max_batch_size = 16);

  ~SimplePIRServer();

  // This is only used when rlwe compression is not used
  std::vector<l_word> hint_gathered_;

  void offline_setup();

  LWEContextPtr<l_word> get_lwe_context(size_t step_idx) {
    return step_idx == 0 ? lwe_ : lwe_small_;
  }

  auto get_a_seed(size_t step_idx) { return a_seed_.at(step_idx); }

  size_t get_num_lwe_plain_word_chunks() {
    return div_ceil<size_t>(sizeof(l_word) * 8 - lwe_round_shift_bits_,
                            sizeof(plain_word) * 8);
  }

  size_t get_num_rlwe_plain_word_chunks() {
    return div_ceil<size_t>(log2_ceil(rlwe_->modulus_) - rlwe_round_shift_bits_,
                            sizeof(plain_word) * 8);
  }

  size_t get_padded_dim(size_t index) { return padded_dims_.at(index); }

  /**
   * @brief Online stage
   *
   * @param query Query matrix
   * @param batch_size batch size of the input query / output response
   * @return Response matrix
   */
  std::vector<std::vector<plain_word>>
  online_stage_synchronized(std::vector<l_word *> &query, size_t batch_size,
                            std::vector<r_word *> &conversion_ciphertexts);
};

} // namespace lwe_ann
