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
#include "rlwe_client.h"
#include "type_utils.h"

namespace lwe_ann {

template <typename uword> class RLWECiphertextPackLD {
  static_assert(is_valid_unsigned_word_v<uword>,
                "uword must be a valid unsigned word type");

public:
  logical_data_t<cudastf::slice<uword>> a_ld_;
  logical_data_t<cudastf::slice<uword>> b_ld_;

  double scale_;
  size_t round_shift_bits_;
  size_t batch_size_;
  bool is_ntt_;

  RLWECiphertextPackLD() = default;
  RLWECiphertextPackLD(logical_data_t<cudastf::slice<uword>> &&a_ld,
                       logical_data_t<cudastf::slice<uword>> &&b_ld,
                       double scale, size_t round_shift_bits, size_t batch_size,
                       bool is_ntt)
      : a_ld_(std::move(a_ld)), b_ld_(std::move(b_ld)), scale_(scale),
        round_shift_bits_(round_shift_bits), batch_size_(batch_size),
        is_ntt_(is_ntt) {}

  // Disable copy but movable
  RLWECiphertextPackLD(RLWECiphertextPackLD &&) = default;
  ~RLWECiphertextPackLD() = default;
};

namespace detail {

template <typename uword>
class RLWEContext_t
    : public RLWEClient<uword>,
      public std::enable_shared_from_this<RLWEContext_t<uword>> {
  static_assert(is_valid_unsigned_word_v<uword>,
                "uword must be a valid unsigned word type");

  template <typename U = uword> using Ct = RLWECiphertextPack<U>;
  template <typename U = uword> using CtLD = RLWECiphertextPackLD<U>;
  using Sec = RLWESecret<uword>;

  static constexpr uint32_t default_grid_dim_ = 2048;
  static constexpr uint32_t default_block_dim_ = 256;

  RLWEContext_t(size_t N, uword modulus, double sigma,
                stf_context_ptr_t cudastf_ctx = nullptr);

  void assert_compatible_ctld(const CtLD<uword> &ctxt1,
                              const CtLD<uword> &ctxt2,
                              bool skip_scale_check = false) const;

public:
  stf_context_ptr_t cudastf_ctx_;

  static inline std::shared_ptr<RLWEContext_t<uword>>
  create(size_t n, uword modulus, double sigma,
         stf_context_ptr_t cudastf_ctx = nullptr) {
    return std::shared_ptr<RLWEContext_t<uword>>(
        new RLWEContext_t<uword>(n, modulus, sigma, cudastf_ctx));
  }

  template <typename U>
  [[nodiscard]] inline CtLD<U> ciphertext_to_ld(const Ct<U> &ctxt) const {
    CtLD<U> res;
    res.scale_ = ctxt.scale_;
    res.round_shift_bits_ = ctxt.round_shift_bits_;
    res.batch_size_ = ctxt.a_data_.batch_size_;
    res.is_ntt_ = ctxt.a_data_.is_ntt_;
    if (ctxt.a_data_.size() != this->N_ * res.batch_size_ ||
        ctxt.b_data_.size() != this->N_ * res.batch_size_ ||
        ctxt.b_data_.batch_size_ != res.batch_size_) {
      throw std::invalid_argument("ciphertext_to_ld failed: size mismatch");
    }
    if (ctxt.a_data_.is_ntt_ != ctxt.b_data_.is_ntt_) {
      throw std::invalid_argument("ciphertext_to_ld failed: is_ntt mismatch");
    }

    res.a_ld_ = vector_to_ld(cudastf_ctx_, ctxt.a_data_);
    res.b_ld_ = vector_to_ld(cudastf_ctx_, ctxt.b_data_);
    return res;
  }

  template <typename U>
  [[nodiscard]] inline CtLD<U> ciphertext_to_ld(Ct<U> &&ctxt) const {
    CtLD<U> res;
    res.scale_ = ctxt.scale_;
    res.round_shift_bits_ = ctxt.round_shift_bits_;
    res.batch_size_ = ctxt.a_data_.batch_size_;
    res.is_ntt_ = ctxt.a_data_.is_ntt_;
    if (ctxt.a_data_.size() != this->N_ * res.batch_size_ ||
        ctxt.b_data_.size() != this->N_ * res.batch_size_ ||
        ctxt.b_data_.batch_size_ != res.batch_size_) {
      throw std::invalid_argument("ciphertext_to_ld failed: size mismatch");
    }
    if (ctxt.a_data_.is_ntt_ != ctxt.b_data_.is_ntt_) {
      throw std::invalid_argument("ciphertext_to_ld failed: is_ntt mismatch");
    }

    res.a_ld_ = vector_to_ld(cudastf_ctx_, std::move(ctxt.a_data_));
    res.b_ld_ = vector_to_ld(cudastf_ctx_, std::move(ctxt.b_data_));
    return res;
  }

  template <typename U>
  [[nodiscard]] inline Ct<U> __ld_to_ciphertext(const CtLD<U> &ld) const {
    Ct<U> res;
    size_t a_size = ld.batch_size_ * this->N_;
    size_t b_size = ld.batch_size_ * this->N_;
    res.a_data_.resize(a_size);
    res.b_data_.resize(b_size);
    __copy_ld_to_host_ptr(cudastf_ctx_, res.a_data_.data(), a_size, ld.a_ld_);
    __copy_ld_to_host_ptr(cudastf_ctx_, res.b_data_.data(), b_size, ld.b_ld_);
    res.scale_ = ld.scale_;
    res.round_shift_bits_ = ld.round_shift_bits_;
    res.a_data_.batch_size_ = ld.batch_size_;
    res.b_data_.batch_size_ = ld.batch_size_;
    res.a_data_.is_ntt_ = ld.is_ntt_;
    res.b_data_.is_ntt_ = ld.is_ntt_;
    return res;
  }

  // We actually don't need these individual primitives for ANN,
  // but they may be useful for general RLWE workload.
  void add(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_in1,
           CtLD<uword> &ctxt_in2) const;
  void add_inplace(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_in2) const;
  void sub(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_a,
           CtLD<uword> &ctxt_b) const;
  void sub_inplace(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_b) const;
  void neg(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt) const;
  void neg_inplace(CtLD<uword> &ctxt_res) const;
};

} // namespace detail

template <typename uword>
using RLWEContextPtr = std::shared_ptr<detail::RLWEContext_t<uword>>;

template <typename uword>
inline RLWEContextPtr<uword>
create_rlwe_context(size_t n, uword modulus, double sigma,
                    stf_context_ptr_t cudastf_ctx = nullptr) {
  return detail::RLWEContext_t<uword>::create(n, modulus, sigma, cudastf_ctx);
}

} // namespace lwe_ann