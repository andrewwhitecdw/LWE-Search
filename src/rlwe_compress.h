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

#include "lwe_cudastf.h"
#include "lwe_server.h"
#include "rlwe_server.h"
#include "type_utils.h"

#include <cmath>

namespace lwe_ann {
namespace detail {

template <typename l_word, typename r_word> class RLWECompress {
private:
  using sr_word = make_signed_t<r_word>;

  LWEContextPtr<l_word> lwe_;
  RLWEContextPtr<r_word> rlwe_;
  const sr_word rlwe_scale_;
  const int device_id_;
  const size_t rlwe_round_shift_bits_;

  static_assert(is_valid_signed_word_v<l_word>,
                "l_word must be a valid signed word type");
  static_assert(is_valid_unsigned_word_v<r_word>,
                "r_word must be a valid unsigned word type");
  static_assert(std::is_same_v<r_word, uint64_t>,
                "Currently, r_word must be uint64_t");
  static constexpr r_word modulus_ = 0x7FFFFFE060001ULL;
  static constexpr uint8_t moduli_index_ = 0;
  static constexpr size_t max_ntt_num_polys_ = 1 << 24;
  // This will be passed to an arbitrary NTT/INTT function
  uint8_t *gpu_moduli_index_ptr_;

  template <bool merged = false>
  void copy_and_ntt(logical_data_t<cudastf::slice<r_word>> &ntt_res,
                    logical_data_t<cudastf::slice<l_word>> &lwe_data,
                    size_t ntt_column_index, size_t current_ntt_column_size,
                    size_t batch_size, bool is_b_data);

  template <bool accumulate, int second_dim_overwrite_ct = -1>
  void gemm_worker(logical_data_t<cudastf::slice<r_word>> &res,
                   logical_data_t<cudastf::slice<r_word>> &a_ntt,
                   logical_data_t<cudastf::slice<r_word>> &conversion_data,
                   size_t batch_size, size_t batch_idx = 0);

  template <typename plain_word>
  void rlwe_reinterpret_worker(
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
      logical_data_t<cudastf::slice<r_word>> &src, size_t batch_size,
      size_t src_idx, bool is_b_data);

  const size_t fldim_;
  const size_t ntt_stride_;
  const size_t ntt_column_size_;

  // A temporary struct to hold the NTT results
  std::vector<logical_data_t<cudastf::slice<r_word>>> a_ntt_results_;

public:
  RLWECompress(LWEContextPtr<l_word> lwe, RLWEContextPtr<r_word> rlwe,
               size_t fldim, size_t ntt_stride, double rlwe_scale,
               size_t rlwe_round_shift_bits = 0, int device_id = 0)
      : lwe_(lwe), rlwe_(rlwe), fldim_(fldim), ntt_stride_(ntt_stride),
        ntt_column_size_(fldim / ntt_stride),
        rlwe_scale_(static_cast<sr_word>(std::llround(rlwe_scale))),
        device_id_(device_id), rlwe_round_shift_bits_(rlwe_round_shift_bits) {
    if (lwe_->cudastf_ctx_ != rlwe_->cudastf_ctx_) {
      throw std::runtime_error(
          "LWE and RLWE CUDASTF contexts must be the same");
    }

    if (fldim_ != ntt_stride_ * ntt_column_size_) {
      throw std::runtime_error(
          "RLWECompress: fldim must be a multiple of ntt_stride");
    }
    if (rlwe_->modulus_ <= (((r_word)1)) << rlwe_round_shift_bits) {
      throw std::runtime_error(
          "RLWECompress: RLWE modulus must be greater than "
          "2^rlwe_round_shift_bits");
    }
    if (rlwe_->modulus_ <= rlwe_scale_) {
      throw std::runtime_error(
          "RLWECompress: RLWE scale must be less than RLWE modulus");
    }

    if (rlwe_->modulus_ != modulus_) {
      throw std::runtime_error("Currently, RLWECompress only support modulus " +
                               std::to_string(modulus_));
    }
    cudastf::cuda_safe_call(cudaSetDevice(device_id_));
    cudastf::cuda_safe_call(
        cudaMalloc(&gpu_moduli_index_ptr_, max_ntt_num_polys_));
    cudastf::cuda_safe_call(
        cudaMemset(gpu_moduli_index_ptr_, moduli_index_, max_ntt_num_polys_));
  }

  ~RLWECompress() {
    cudastf::cuda_safe_call(cudaFree(gpu_moduli_index_ptr_));
  }

  /**
   * @brief Compute the NTT of a_data and store the results in a_ntt_results_.
   *        Data must have the following dimensions:
   *        - a_data: (lwe_n, fldim = ntt_stride * ntt_column_size)
   *        - lwe_n: lwe dimension, must be a multiple of 8
   *        - fldim: must be a multiple of ntt_stride (a multiple of 64)
   *
   * @param a_data The a part of the data produced by SimplePIRServer
   * @param fldim corresponds to dim_rest in SimplePIRServer, multiple of
   * ntt_stride
   * @param ntt_stride The stride to perform NTT on a_data, corresponds to
   * (dim_rest / dim_last) for a multi-dimensional SimplePIR, multiple of 64
   */
  void prepare_a_ntt(logical_data_t<cudastf::slice<l_word>> &a_data);

  /**
   * @brief Multiply the conversion ciphertexts and store the results in res_a
   * and res_b.
   *
   * @param res_a The result of the multiplication of conversion ciphertexts
   * @param res_b The result of the multiplication of conversion ciphertexts
   * @param conversion_a The conversion ciphertexts for a_data
   * @param conversion_b The conversion ciphertexts for b_data
   * @param lwe_b The b part of the data produced by SimplePIRServer
   * @param lwe_batch_size The batch size of LWE encryption
   */
  void multiply_conversion_ciphertexts(
      std::vector<logical_data_t<cudastf::slice<r_word>>> &res_a,
      std::vector<logical_data_t<cudastf::slice<r_word>>> &res_b,
      logical_data_t<cudastf::slice<r_word>> &conversion_a,
      logical_data_t<cudastf::slice<r_word>> &conversion_b,
      logical_data_t<cudastf::slice<l_word>> &lwe_b, size_t lwe_batch_size);

  template <typename plain_word>
  void multiply_conversion_ciphertexts_with_reinterpret(
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
      logical_data_t<cudastf::slice<r_word>> &conversion_a,
      logical_data_t<cudastf::slice<r_word>> &conversion_b,
      logical_data_t<cudastf::slice<l_word>> &lwe_b, size_t lwe_batch_size);

  template <typename plain_word>
  void do_everything_per_batch(
      std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
      logical_data_t<cudastf::slice<l_word>> &merged_data,
      logical_data_t<cudastf::slice<r_word>> &conversion_a,
      logical_data_t<cudastf::slice<r_word>> &conversion_b, size_t batch_idx);

  /**
   * @brief Round the results of multiply_conversion_ciphertexts
   *
   * @param res Rounded results (in the same data layout as long_res)
   * @param long_res The result of multiply_conversion_ciphertexts (a)
   * @param batch_size The batch size of LWE encryption
   */
  // template <typename r_short = r_word>
  // void
  // rlwe_round(std::vector<logical_data_t<cudastf::slice<r_short>>> &res,
  //            std::vector<logical_data_t<cudastf::slice<r_word>>> &&long_res,
  //            size_t batch_size);

  /**
   * @brief Perform RLWE rounding and reorganize the data layout for
   *        subsequent dimension reduction steps for PIR.
   *
   * @param res The reinterpreted results, concatenated and reordered
   * @param long_res_a The result of multiply_conversion_ciphertexts (a)
   * @param long_res_b The result of multiply_conversion_ciphertexts (b)
   * @param batch_size The batch size of LWE encryption
   */
  // template <typename plain_word>
  // void reinterpret_results(
  //     std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
  //     std::vector<logical_data_t<cudastf::slice<r_word>>> &&long_res_a,
  //     std::vector<logical_data_t<cudastf::slice<r_word>>> &&long_res_b,
  //     size_t batch_size);

  template <typename plain_word> size_t get_num_plain_word_chunks() const {
    return div_ceil<size_t>(log2_ceil(rlwe_->modulus_) - rlwe_round_shift_bits_,
                            sizeof(plain_word) * 8);
  }

  template <typename plain_word>
  size_t get_reinterpreted_size(size_t batch_size = 1) const {
    size_t vector_size = div_ceil<size_t>(ntt_column_size_, rlwe_->N_);
    size_t num_plain_word_chunks = get_num_plain_word_chunks<plain_word>();
    return vector_size * num_plain_word_chunks * rlwe_->N_ * ntt_stride_ *
           batch_size * 2;
  }

  /**
   * @brief Perform NTT in place. All polynomials should be in a contiguous
   * space and the coefficients for each polynomial should be adjacent.
   *
   * @param ntt_inout The input and output data
   * @param num_polys The number of polynomials
   */
  void ntt_inplace(logical_data_t<cudastf::slice<r_word>> &ntt_inout,
                   size_t num_polys);

  /**
   * @brief Perform INTT in place. All polynomials should be in a contiguous
   * space and the coefficients for each polynomial should be adjacent.
   *
   * @param intt_inout The input and output data
   * @param num_polys The number of polynomials
   */
  void inverse_ntt_inplace(logical_data_t<cudastf::slice<r_word>> &intt_inout,
                           size_t num_polys);
};
} // namespace detail

template <typename l_word, typename r_word>
using RLWECompressPtr = std::shared_ptr<detail::RLWECompress<l_word, r_word>>;

template <typename l_word, typename r_word>
RLWECompressPtr<l_word, r_word>
create_rlwe_compress(LWEContextPtr<l_word> lwe, RLWEContextPtr<r_word> rlwe,
                     size_t fldim, size_t ntt_stride, double rlwe_scale,
                     size_t rlwe_round_shift_bits = 0, int device_id = 0) {
  return std::make_shared<detail::RLWECompress<l_word, r_word>>(
      lwe, rlwe, fldim, ntt_stride, rlwe_scale, rlwe_round_shift_bits,
      device_id);
}

} // namespace lwe_ann