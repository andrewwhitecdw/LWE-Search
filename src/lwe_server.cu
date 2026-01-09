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
 
#include "lwe_server.h"

#include "../rlwe_kernels/constants.h"
#include <stdexcept>

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
  __get_grid_dim<LWEContext_t<word>::default_block_dim_,                       \
                 LWEContext_t<word>::default_grid_dim_>(size)

namespace lwe_ann {
namespace detail {

template <typename word>
LWEContext_t<word>::LWEContext_t(size_t n, double sigma,
                                 stf_context_ptr_t cudastf_ctx)
    : LWEClient<word>(n, sigma) {
  // If not given, create a new CUDASTF context
  if (cudastf_ctx == nullptr) {
    // Initialize CUDASTF context
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
}

template <typename word>
void LWEContext_t<word>::assert_compatible_ctld(const CtLD &ctxt1,
                                                const CtLD &ctxt2,
                                                bool skip_scale_check) const {
  if (ctxt1.ptxt_len_ != ctxt2.ptxt_len_ ||
      ctxt1.batch_size_ != ctxt2.batch_size_) {
    throw std::invalid_argument("assert_compatible_ctld failed: size mismatch");
  }
  if (ctxt1.round_shift_bits_ != ctxt2.round_shift_bits_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: round_shift_bits mismatch");
  }
  if (skip_scale_check)
    return;
  if (ctxt1.scale_ != ctxt2.scale_) {
    throw std::invalid_argument(
        "assert_compatible_ctld failed: scale mismatch");
  }
}

template <typename word>
__global__ void lwe_add_kernel_(cudastf::slice<word> res,
                                cudastf::slice<const word> a,
                                cudastf::slice<const word> b) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) = a(i) + b(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_add_inplace_kernel_(cudastf::slice<word> res,
                                        cudastf::slice<const word> in2) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) += in2(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_sub_kernel_(cudastf::slice<word> res,
                                cudastf::slice<const word> a,
                                cudastf::slice<const word> b) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) = a(i) - b(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_sub_inplace_kernel_(cudastf::slice<word> res,
                                        cudastf::slice<const word> b) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) -= b(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_neg_kernel_(cudastf::slice<word> res,
                                cudastf::slice<const word> a) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) = -a(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_neg_inplace_kernel_(cudastf::slice<word> res) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) = -res(i);
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_mult_constant_kernel_(cudastf::slice<word> res,
                                          cudastf::slice<const word> a,
                                          word constant) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) = a(i) * constant;
  STRIDED_LOOP_END;
}

template <typename word>
__global__ void lwe_mult_constant_inplace_kernel_(cudastf::slice<word> res,
                                                  word constant) {
  STRIDED_LOOP_START(res.size(), i);
  res(i) *= constant;
  STRIDED_LOOP_END;
}

template <typename word>
void LWEContext_t<word>::add(CtLD &ctxt_res, CtLD &ctxt_in1,
                             CtLD &ctxt_in2) const {
  // Check size of input
  if (&ctxt_res == &ctxt_in1) {
    add_inplace(ctxt_res, ctxt_in2);
    return;
  } else if (&ctxt_res == &ctxt_in2) {
    add_inplace(ctxt_res, ctxt_in1);
  }
  assert_compatible_ctld(ctxt_in1, ctxt_in2);

  auto kernel_desc_func = [](cudastf::slice<word> res,
                             cudastf::slice<const word> in1,
                             cudastf::slice<const word> in2) {
    if (res.size() != in1.size() || res.size() != in2.size()) {
      throw std::invalid_argument("add failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{
        lwe_add_kernel_<word>, grid_dim, default_block_dim_, 0, res, in1, in2};
  };

  cudastf_ctx_
          ->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in1.a_ld_.read(),
                        ctxt_in2.a_ld_.read())
          .set_symbol("LWE_Add_A")
          ->*kernel_desc_func;
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in1.b_ld_.read(),
                        ctxt_in2.b_ld_.read())
          .set_symbol("LWE_Add_B")
          ->*kernel_desc_func;
  ctxt_res.ptxt_len_ = ctxt_in1.ptxt_len_;
  ctxt_res.batch_size_ = ctxt_in1.batch_size_;
  ctxt_res.scale_ = ctxt_in1.scale_;
  ctxt_res.round_shift_bits_ = ctxt_in1.round_shift_bits_;
}

template <typename word>
void LWEContext_t<word>::add_inplace(CtLD &ctxt_res, CtLD &ctxt_in2) const {
  // Check size of input
  assert_compatible_ctld(ctxt_res, ctxt_in2);

  auto kernel_desc_func = [](cudastf::slice<word> res,
                             cudastf::slice<const word> in2) {
    if (res.size() != in2.size()) {
      throw std::invalid_argument("add failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{lwe_add_inplace_kernel_<word>,
                                     grid_dim,
                                     default_block_dim_,
                                     0,
                                     res,
                                     in2};
  };

  // Just launch two kernels
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw(), ctxt_in2.a_ld_.read())
          .set_symbol("LWE_Add_Inplace_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw(), ctxt_in2.b_ld_.read())
          .set_symbol("LWE_Add_Inplace_B")
          ->*kernel_desc_func;
}

template <typename word>
void LWEContext_t<word>::sub(CtLD &ctxt_res, CtLD &ctxt_in1,
                             CtLD &ctxt_in2) const {
  // Check size of input
  if (&ctxt_res == &ctxt_in1) {
    sub_inplace(ctxt_res, ctxt_in2);
    return;
  } else if (&ctxt_res == &ctxt_in2) {
    sub_inplace(ctxt_res, ctxt_in1);
  }
  assert_compatible_ctld(ctxt_res, ctxt_in2);

  auto kernel_desc_func = [](cudastf::slice<word> res,
                             cudastf::slice<const word> in1,
                             cudastf::slice<const word> in2) {
    if (res.size() != in1.size() || res.size() != in2.size()) {
      throw std::invalid_argument("sub failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{
        lwe_sub_kernel_<word>, grid_dim, default_block_dim_, 0, res, in1, in2};
  };

  cudastf_ctx_
          ->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in1.a_ld_.read(),
                        ctxt_in2.a_ld_.read())
          .set_symbol("LWE_Sub_A")
          ->*kernel_desc_func;
  cudastf_ctx_
          ->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in1.b_ld_.read(),
                        ctxt_in2.b_ld_.read())
          .set_symbol("LWE_Sub_B")
          ->*kernel_desc_func;
  ctxt_res.scale_ = ctxt_in1.scale_;
  ctxt_res.ptxt_len_ = ctxt_in1.ptxt_len_;
  ctxt_res.batch_size_ = ctxt_in1.batch_size_;
  ctxt_res.round_shift_bits_ = ctxt_in1.round_shift_bits_;
}

template <typename word>
void LWEContext_t<word>::sub_inplace(CtLD &ctxt_res, CtLD &ctxt_in2) const {
  // Check size of input
  assert_compatible_ctld(ctxt_res, ctxt_in2);

  auto kernel_desc_func = [](cudastf::slice<word> res,
                             cudastf::slice<const word> in2) {
    if (res.size() != in2.size()) {
      throw std::invalid_argument("sub failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{lwe_sub_inplace_kernel_<word>,
                                     grid_dim,
                                     default_block_dim_,
                                     0,
                                     res,
                                     in2};
  };

  // Just launch two kernels
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw(), ctxt_in2.a_ld_.read())
          .set_symbol("LWE_Sub_Inplace_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw(), ctxt_in2.b_ld_.read())
          .set_symbol("LWE_Sub_Inplace_B")
          ->*kernel_desc_func;
}

template <typename word>
void LWEContext_t<word>::neg(CtLD &ctxt_res, CtLD &ctxt_in) const {
  // Check size of input
  if (&ctxt_res == &ctxt_in) {
    neg_inplace(ctxt_res);
    return;
  }

  auto kernel_desc_func = [](cudastf::slice<word> res,
                             cudastf::slice<const word> in) {
    if (res.size() != in.size()) {
      throw std::invalid_argument("neg failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{
        lwe_neg_kernel_<word>, grid_dim, default_block_dim_, 0, res, in};
  };

  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in.a_ld_.read())
          .set_symbol("LWE_Neg_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in.b_ld_.read())
          .set_symbol("LWE_Neg_B")
          ->*kernel_desc_func;
  ctxt_res.scale_ = ctxt_in.scale_;
  ctxt_res.ptxt_len_ = ctxt_in.ptxt_len_;
  ctxt_res.batch_size_ = ctxt_in.batch_size_;
  ctxt_res.round_shift_bits_ = ctxt_in.round_shift_bits_;
}

template <typename word>
void LWEContext_t<word>::neg_inplace(CtLD &ctxt_res) const {
  auto kernel_desc_func = [](cudastf::slice<word> res) {
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{lwe_neg_inplace_kernel_<word>, grid_dim,
                                     default_block_dim_, 0, res};
  };

  // Just launch two kernels
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw()).set_symbol("LWE_Neg_Inplace_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw()).set_symbol("LWE_Neg_Inplace_B")
          ->*kernel_desc_func;
}

template <typename word>
void LWEContext_t<word>::mult_constant(CtLD &ctxt_res, CtLD &ctxt_in,
                                       word constant) const {
  // Check size of input
  if (&ctxt_res == &ctxt_in) {
    mult_constant_inplace(ctxt_res, constant);
    return;
  }

  auto kernel_desc_func = [constant = constant](cudastf::slice<word> res,
                                                cudastf::slice<const word> in) {
    if (res.size() != in.size()) {
      throw std::invalid_argument("mult_constant failed: size mismatch");
    }
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{lwe_mult_constant_kernel_<word>,
                                     grid_dim,
                                     default_block_dim_,
                                     0,
                                     res,
                                     in,
                                     constant};
  };

  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.write(), ctxt_in.a_ld_.read())
          .set_symbol("LWE_Mult_Constant_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.write(), ctxt_in.b_ld_.read())
          .set_symbol("LWE_Mult_Constant_B")
          ->*kernel_desc_func;
  ctxt_res.scale_ = ctxt_in.scale_;
  ctxt_res.ptxt_len_ = ctxt_in.ptxt_len_;
  ctxt_res.batch_size_ = ctxt_in.batch_size_;
  ctxt_res.round_shift_bits_ = ctxt_in.round_shift_bits_;
}

template <typename word>
void LWEContext_t<word>::mult_constant_inplace(CtLD &ctxt_res,
                                               word constant) const {
  auto kernel_desc_func = [constant = constant](cudastf::slice<word> res) {
    uint32_t grid_dim = GET_GRID_DIM(res.size());
    return cudastf::cuda_kernel_desc{lwe_mult_constant_inplace_kernel_<word>,
                                     grid_dim,
                                     default_block_dim_,
                                     0,
                                     res,
                                     constant};
  };

  // Just launch two kernels
  cudastf_ctx_->cuda_kernel(ctxt_res.a_ld_.rw())
          .set_symbol("LWE_Mult_Constant_Inplace_A")
          ->*kernel_desc_func;
  cudastf_ctx_->cuda_kernel(ctxt_res.b_ld_.rw())
          .set_symbol("LWE_Mult_Constant_Inplace_B")
          ->*kernel_desc_func;
}

template class LWEContext_t<int32_t>;

} // namespace detail

template LWEContextPtr<int32_t>
create_lwe_context<int32_t>(size_t n, double sigma,
                            stf_context_ptr_t cudastf_ctx);

} // namespace lwe_ann
