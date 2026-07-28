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
 
#include "rlwe_server.h"

#include "../rlwe_kernels/constants.h"
#include "../rlwe_kernels/ntt_twiddles.h"

namespace {

template <uint32_t default_block_dim, uint32_t default_grid_dim>
uint32_t __get_grid_dim(size_t size) {
  if (size > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("size is too large");
  }
  uint32_t grid_dim =
      (static_cast<uint32_t>(size) + default_block_dim - 1) / default_block_dim;
  if (grid_dim > default_grid_dim) {
    grid_dim = default_grid_dim;
  }
  return grid_dim;
}

} // namespace

#define GET_GRID_DIM(size)                                                     \
  __get_grid_dim<RLWEContext_t<uword>::default_block_dim_,                     \
                 RLWEContext_t<uword>::default_grid_dim_>(size)

namespace lwe_ann {
namespace detail {

template <typename uword>
RLWEContext_t<uword>::RLWEContext_t(size_t N, uword modulus, double sigma,
                                    stf_context_ptr_t cudastf_ctx)
    : RLWEClient<uword>(N, modulus, sigma) {
  if (cudastf_ctx == nullptr) {
    auto deleter = [](stf_context_t *p) {
      if (p) {
        p->finalize();
        delete p;
      }
    };
    std::shared_ptr<stf_context_t> pctx(new stf_context_t(), deleter);
    cudastf_ctx_ = pctx;
  } else {
    cudastf_ctx_ = cudastf_ctx;
  }

  // Initialize NTT twiddle factors (lightweight, no full modulus chain)
  ntt_twiddles::init_twiddle_factors(static_cast<uint64_t>(modulus), N, 0);
}

template <typename uword>
void RLWEContext_t<uword>::assert_compatible_ctld(const CtLD<uword> &ctxt1,
                                                  const CtLD<uword> &ctxt2,
                                                  bool skip_scale_check) const {
  if (ctxt1.batch_size_ != ctxt2.batch_size_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: batch_size mismatch");
  }
  if (ctxt1.round_shift_bits_ != ctxt2.round_shift_bits_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: round_shift_bits mismatch");
  }
  if (ctxt1.is_ntt_ != ctxt2.is_ntt_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: is_ntt mismatch");
  }
  if (skip_scale_check)
    return;
  if (ctxt1.scale_ != ctxt2.scale_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: scale mismatch");
  }
}

template <typename uword>
__global__ void rlwe_add_kernel_(cudastf::slice<uword> res,
                                 cudastf::slice<const uword> a,
                                 cudastf::slice<const uword> b, uword modulus) {
  STRIDED_LOOP_START(res.size(), i);
  uword sum = a(i) + b(i);
  res(i) = sum - (sum >= modulus) * modulus;
  STRIDED_LOOP_END;
}

template <typename uword>
__global__ void rlwe_add_inplace_kernel_(cudastf::slice<uword> res,
                                         cudastf::slice<const uword> in2,
                                         uword modulus) {
  STRIDED_LOOP_START(res.size(), i);
  uword sum = res(i) + in2(i);
  res(i) = sum - (sum >= modulus) * modulus;
  STRIDED_LOOP_END;
}

template <typename uword>
__global__ void rlwe_sub_kernel_(cudastf::slice<uword> res,
                                 cudastf::slice<const uword> a,
                                 cudastf::slice<const uword> b, uword modulus) {
  using sword = make_signed_t<uword>;

  STRIDED_LOOP_START(res.size(), i);
  sword diff = static_cast<sword>(a(i)) - static_cast<sword>(b(i));
  res(i) = static_cast<uword>(diff + (diff < 0) * static_cast<sword>(modulus));
  STRIDED_LOOP_END;
}

template <typename uword>
__global__ void rlwe_sub_inplace_kernel_(cudastf::slice<uword> res,
                                         cudastf::slice<const uword> b,
                                         uword modulus) {
  using sword = make_signed_t<uword>;

  STRIDED_LOOP_START(res.size(), i);
  sword diff = static_cast<sword>(res(i)) - static_cast<sword>(b(i));
  res(i) = static_cast<uword>(diff + (diff < 0) * static_cast<sword>(modulus));
  STRIDED_LOOP_END;
}

template <typename uword>
__global__ void rlwe_neg_kernel_(cudastf::slice<uword> res,
                                 cudastf::slice<const uword> a, uword modulus) {
  STRIDED_LOOP_START(res.size(), i);
  uword a_i = a(i);
  res(i) = (a_i != 0) * modulus - a_i;
  STRIDED_LOOP_END;
}

template <typename uword>
__global__ void rlwe_neg_inplace_kernel_(cudastf::slice<uword> res,
                                         uword modulus) {
  STRIDED_LOOP_START(res.size(), i);
  uword res_i = res(i);
  res(i) = (res_i != 0) * modulus - res_i;
  STRIDED_LOOP_END;
}

template <typename uword>
void RLWEContext_t<uword>::add(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_in1,
                               CtLD<uword> &ctxt_in2) const {
  if (&ctxt_res == &ctxt_in1) {
    add_inplace(ctxt_res, ctxt_in2);
    return;
  } else if (&ctxt_res == &ctxt_in2) {
    add_inplace(ctxt_res, ctxt_in1);
    return;
  }
  assert_compatible_ctld(ctxt_in1, ctxt_in2);

  auto kernel_desc_func =
      [mod = this->modulus_](cudastf::slice<uword> res,
                             cudastf::slice<const uword> in1,
                             cudastf::slice<const uword> in2) {
        if (res.size() != in1.size() || res.size() != in2.size()) {
          throw std::invalid_argument("add failed: size mismatch");
        }
        uint32_t grid_dim = GET_GRID_DIM(res.size());
        return cudastf::cuda_kernel_desc{rlwe_add_kernel_<uword>,
                                         grid_dim,
                                         default_block_dim_,
                                         0,
                                         res,
                                         in1,
                                         in2,
                                         mod};
      };
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in1.a_ld_.read(),
                        ctxt_in2.a_ld_.read())
          .set_symbol("RLWE_Add_A")
          ->*kernel_desc_func;
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in1.b_ld_.read(),
                        ctxt_in2.b_ld_.read())
          .set_symbol("RLWE_Add_B")
          ->*kernel_desc_func;
  ctxt_res.batch_size_ = ctxt_in1.batch_size_;
  ctxt_res.is_ntt_ = ctxt_in1.is_ntt_;
  ctxt_res.scale_ = ctxt_in1.scale_;
  ctxt_res.round_shift_bits_ = ctxt_in1.round_shift_bits_;
}

template <typename uword>
void RLWEContext_t<uword>::add_inplace(CtLD<uword> &ctxt_res,
                                       CtLD<uword> &ctxt_in2) const {
  assert_compatible_ctld(ctxt_res, ctxt_in2);

  auto kernel_desc_func =
      [mod = this->modulus_](cudastf::slice<uword> res,
                             cudastf::slice<const uword> in2) {
        if (res.size() != in2.size()) {
          throw std::invalid_argument("add failed: size mismatch");
        }
        uint32_t grid_dim = GET_GRID_DIM(res.size());
        return cudastf::cuda_kernel_desc{rlwe_add_inplace_kernel_<uword>,
                                         grid_dim,
                                         default_block_dim_,
                                         0,
                                         res,
                                         in2,
                                         mod};
      };
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw(), ctxt_in2.a_ld_.read())
          .set_symbol("RLWE_Add_A_Inplace")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw(), ctxt_in2.b_ld_.read())
          .set_symbol("RLWE_Add_B_Inplace")
          ->*kernel_desc_func;
}

template <typename uword>
void RLWEContext_t<uword>::sub(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt_in1,
                               CtLD<uword> &ctxt_in2) const {
  if (&ctxt_res == &ctxt_in1) {
    sub_inplace(ctxt_res, ctxt_in2);
    return;
  } else if (&ctxt_res == &ctxt_in2) {
    sub_inplace(ctxt_res, ctxt_in1);
  }
  assert_compatible_ctld(ctxt_in1, ctxt_in2);

  auto kernel_desc_func =
      [mod = this->modulus_](cudastf::slice<uword> res,
                             cudastf::slice<const uword> in1,
                             cudastf::slice<const uword> in2) {
        if (res.size() != in1.size() || res.size() != in2.size()) {
          throw std::invalid_argument("sub failed: size mismatch");
        }
        uint32_t grid_dim = GET_GRID_DIM(res.size());
        return cudastf::cuda_kernel_desc{rlwe_sub_kernel_<uword>,
                                         grid_dim,
                                         default_block_dim_,
                                         0,
                                         res,
                                         in1,
                                         in2,
                                         mod};
      };
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in1.a_ld_.read(),
                        ctxt_in2.a_ld_.read())
          .set_symbol("RLWE_Sub_A")
          ->*kernel_desc_func;
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in1.b_ld_.read(),
                        ctxt_in2.b_ld_.read())
          .set_symbol("RLWE_Sub_B")
          ->*kernel_desc_func;

  ctxt_res.batch_size_ = ctxt_in1.batch_size_;
  ctxt_res.is_ntt_ = ctxt_in1.is_ntt_;
  ctxt_res.scale_ = ctxt_in1.scale_;
  ctxt_res.round_shift_bits_ = ctxt_in1.round_shift_bits_;
}

template <typename uword>
void RLWEContext_t<uword>::sub_inplace(CtLD<uword> &ctxt_res,
                                       CtLD<uword> &ctxt_in2) const {
  assert_compatible_ctld(ctxt_res, ctxt_in2);

  auto kernel_desc_func =
      [mod = this->modulus_](cudastf::slice<uword> res,
                             cudastf::slice<const uword> in2) {
        if (res.size() != in2.size()) {
          throw std::invalid_argument("sub failed: size mismatch");
        }
        uint32_t grid_dim = GET_GRID_DIM(res.size());
        return cudastf::cuda_kernel_desc{rlwe_sub_inplace_kernel_<uword>,
                                         grid_dim,
                                         default_block_dim_,
                                         0,
                                         res,
                                         in2,
                                         mod};
      };
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw(), ctxt_in2.a_ld_.read())
          .set_symbol("RLWE_Sub_A_Inplace")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw(), ctxt_in2.b_ld_.read())
          .set_symbol("RLWE_Sub_B_Inplace")
          ->*kernel_desc_func;
}

template <typename uword>
void RLWEContext_t<uword>::neg(CtLD<uword> &ctxt_res, CtLD<uword> &ctxt) const {
  if (&ctxt_res == &ctxt) {
    neg_inplace(ctxt_res);
    return;
  }

  auto kernel_desc_func = [mod =
                               this->modulus_](cudastf::slice<uword> res,
                                               cudastf::slice<const uword> a) {
    if (res.size() != a.size()) {
      throw std::invalid_argument("neg failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{
        rlwe_neg_kernel_<uword>, grid_dim, default_block_dim_, 0, res, a, mod};
  };
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.write(), ctxt.a_ld_.read())
          .set_symbol("RLWE_Neg_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.write(), ctxt.b_ld_.read())
          .set_symbol("RLWE_Neg_B")
          ->*kernel_desc_func;

  ctxt_res.batch_size_ = ctxt.batch_size_;
  ctxt_res.is_ntt_ = ctxt.is_ntt_;
  ctxt_res.scale_ = ctxt.scale_;
  ctxt_res.round_shift_bits_ = ctxt.round_shift_bits_;
}

template <typename uword>
void RLWEContext_t<uword>::neg_inplace(CtLD<uword> &ctxt_res) const {
  auto kernel_desc_func = [mod = this->modulus_](cudastf::slice<uword> res) {
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{rlwe_neg_inplace_kernel_<uword>,
                                     grid_dim,
                                     default_block_dim_,
                                     0,
                                     res,
                                     mod};
  };
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw())
          .set_symbol("RLWE_Neg_A_Inplace")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw())
          .set_symbol("RLWE_Neg_B_Inplace")
          ->*kernel_desc_func;
}

template class RLWEContext_t<uint64_t>;

} // namespace detail
} // namespace lwe_ann
