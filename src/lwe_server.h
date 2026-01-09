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

#include <limits>
#include <memory>
#include <vector>

#include "lwe_client.h"
#include "lwe_cudastf.h"
#include "type_utils.h"

namespace lwe_ann {

template <typename word> class LWECiphertextPackLD {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");

public:
  logical_data_t<cudastf::slice<word>> a_ld_;
  logical_data_t<cudastf::slice<word>> b_ld_;

  double scale_;
  size_t round_shift_bits_;
  size_t ptxt_len_;
  size_t batch_size_;

  LWECiphertextPackLD() = default;

  // Disable copy but movable
  LWECiphertextPackLD(LWECiphertextPackLD &&) = default;
  ~LWECiphertextPackLD() = default;
};

namespace detail {

template <typename word>
class LWEContext_t : public LWEClient<word>,
                     public std::enable_shared_from_this<LWEContext_t<word>> {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");

private:
  using Ct = LWECiphertextPack<word>;
  using CtLD = LWECiphertextPackLD<word>;
  using Sec = LWESecretPack<word>;

  static constexpr uint32_t default_grid_dim_ = 2048;
  static constexpr uint32_t default_block_dim_ = 256;

  LWEContext_t(size_t n, double sigma, stf_context_ptr_t cudastf_ctx = nullptr);

  void assert_compatible_ctld(const CtLD &ctxt1, const CtLD &ctxt2,
                              bool skip_scale_check = false) const;

public:
  stf_context_ptr_t cudastf_ctx_;

  static inline std::shared_ptr<LWEContext_t<word>>
  create(size_t n, double sigma, stf_context_ptr_t cudastf_ctx = nullptr) {
    return std::shared_ptr<LWEContext_t<word>>(
        new LWEContext_t<word>(n, sigma, cudastf_ctx));
  }

  [[nodiscard]] inline CtLD ciphertext_to_ld(const Ct &ctxt) const {
    CtLD res;
    res.scale_ = ctxt.scale_;
    res.round_shift_bits_ = ctxt.round_shift_bits_;
    res.ptxt_len_ = ctxt.a_data_.size() / this->n_;
    res.batch_size_ = ctxt.b_data_.size() / res.ptxt_len_;
    if (res.ptxt_len_ * this->n_ != ctxt.a_data_.size() ||
        res.ptxt_len_ * res.batch_size_ != ctxt.b_data_.size()) {
      throw std::invalid_argument("ciphertext_to_ld failed: size mismatch");
    }
    res.a_ld_ = vector_to_ld(cudastf_ctx_, ctxt.a_data_);
    res.b_ld_ = vector_to_ld(cudastf_ctx_, ctxt.b_data_);
    return res;
  }

  [[nodiscard]] inline CtLD ciphertext_to_ld(Ct &&ctxt) const {
    CtLD res;
    res.scale_ = ctxt.scale_;
    res.round_shift_bits_ = ctxt.round_shift_bits_;
    res.ptxt_len_ = ctxt.a_data_.size() / this->n_;
    res.batch_size_ = ctxt.b_data_.size() / res.ptxt_len_;
    if (res.ptxt_len_ * this->n_ != ctxt.a_data_.size() ||
        res.ptxt_len_ * res.batch_size_ != ctxt.b_data_.size()) {
      throw std::invalid_argument("ciphertext_to_ld failed: size mismatch");
    }
    res.a_ld_ = vector_to_ld(cudastf_ctx_, std::move(ctxt.a_data_));
    res.b_ld_ = vector_to_ld(cudastf_ctx_, std::move(ctxt.b_data_));
    return res;
  }

  // Warning: This function has synchronization side effect.
  [[nodiscard]] inline Ct __ld_to_ciphertext(const CtLD &ld) const {
    Ct res;
    size_t a_size = ld.ptxt_len_ * this->n_;
    size_t b_size = ld.ptxt_len_ * ld.batch_size_;
    res.a_data_.resize(a_size);
    res.b_data_.resize(b_size);
    __copy_ld_to_host_ptr(cudastf_ctx_, res.a_data_.data(), a_size, ld.a_ld_);
    __copy_ld_to_host_ptr(cudastf_ctx_, res.b_data_.data(), b_size, ld.b_ld_);
    res.scale_ = ld.scale_;
    res.round_shift_bits_ = ld.round_shift_bits_;
    return res;
  }

  // We actually don't need these individual primitives for ANN,
  // but they may be useful for general LWE workload.
  void add(CtLD &ctxt_res, CtLD &ctxt_in1, CtLD &ctxt_in2) const;
  void add_inplace(CtLD &ctxt_res, CtLD &ctxt_in2) const;
  void sub(CtLD &ctxt_res, CtLD &ctxt_a, CtLD &ctxt_b) const;
  void sub_inplace(CtLD &ctxt_res, CtLD &ctxt_b) const;
  void neg(CtLD &ctxt_res, CtLD &ctxt) const;
  void neg_inplace(CtLD &ctxt_res) const;
  void mult_constant(CtLD &ctxt_res, CtLD &ctxt, word constant) const;
  void mult_constant_inplace(CtLD &ctxt_res, word constant) const;
};

} // namespace detail

template <typename word>
using LWEContextPtr = std::shared_ptr<detail::LWEContext_t<word>>;

template <typename word>
inline LWEContextPtr<word>
create_lwe_context(size_t n, double sigma,
                   stf_context_ptr_t cudastf_ctx = nullptr) {
  return detail::LWEContext_t<word>::create(n, sigma, cudastf_ctx);
}

} // namespace lwe_ann
