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
 
#include "arith_utils.h"
#include "gemm.h"
#include "simplepir.h"

#include <functional>
#include <limits>
#include <numeric>

namespace lwe_ann {

template <typename T>
void print_matrix(T *matrix, size_t M, size_t N, bool row_major = true) {
  std::cout << "[" << std::endl;
  for (size_t i = 0; i < M; i++) {
    std::cout << "[";
    for (size_t j = 0; j < N; j++) {
      auto value = row_major ? matrix[i * N + j] : matrix[j * M + i];
      std::cout << zero_extend<uint64_t>(value);
      if (j != N - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";
    if (i != M - 1) {
      std::cout << ",";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

template <typename word>
__global__ void copy1_kernel(word *dst, const word *src,
                             unsigned int copy_size) {
  constexpr unsigned int num_words_per_128b = 16 / sizeof(word);
  for (unsigned int idx = threadIdx.x * num_words_per_128b; idx < copy_size;
       idx += blockDim.x * num_words_per_128b) {
    if (idx + num_words_per_128b > copy_size) {
      while (idx < copy_size) {
        dst[idx] = src[idx];
        idx += 1;
      }
    } else {
      *(reinterpret_cast<int4 *>(dst + idx)) =
          *(reinterpret_cast<const int4 *>(src + idx));
    }
  }
}

template <typename word>
__global__ void copy2_kernel(word *dst_a, word *dst_b, const word *src_a,
                             const word *src_b, unsigned int copy_size) {
  constexpr unsigned int num_words_per_128b = 16 / sizeof(word);
  for (unsigned int idx = threadIdx.x * num_words_per_128b; idx < copy_size;
       idx += blockDim.x * num_words_per_128b) {
    if (idx + num_words_per_128b > copy_size) {
      while (idx < copy_size) {
        dst_a[idx] = src_a[idx];
        dst_b[idx] = src_b[idx];
        idx += 1;
      }
    } else {
      *(reinterpret_cast<int4 *>(dst_a + idx)) =
          *(reinterpret_cast<const int4 *>(src_a + idx));
      *(reinterpret_cast<int4 *>(dst_b + idx)) =
          *(reinterpret_cast<const int4 *>(src_b + idx));
    }
  }
}

template <typename l_word, typename r_word, typename plain_word>
SimplePIRServer<l_word, r_word, plain_word>::~SimplePIRServer() {
  for (auto &device_db_ptr : device_db_ptrs_) {
    cudastf::cuda_safe_call(cudaFree(device_db_ptr));
  }
}

template <typename l_word, typename r_word, typename plain_word>
SimplePIRServer<l_word, r_word, plain_word>::SimplePIRServer(
    std::vector<plain_word> &&db, const std::vector<size_t> &dims,
    LWEContextPtr<l_word> lwe, size_t lwe_round_shift_bits,
    RLWEContextPtr<r_word> rlwe, size_t rlwe_round_shift_bits,
    const std::vector<int> &device_ids, size_t max_batch_size)
    : lwe_{lwe}, rlwe_{rlwe}, orig_dims_{dims},
      lwe_round_shift_bits_{lwe_round_shift_bits},
      rlwe_round_shift_bits_{rlwe_round_shift_bits}, device_ids_{device_ids},
      max_batch_size_{pad_by<size_t>(max_batch_size, tile_m_min_)} {
  // Do some checks
  if (orig_dims_.size() != 2 && orig_dims_.size() != 3) {
    // The main complication is related to how padding is done in
    // between each dimension reduction steps.
    throw std::invalid_argument(
        "Currently, only 2D and 3D constructions are supported");
  }
  if (lwe_ == nullptr) {
    throw std::invalid_argument("lwe_ must be provided");
  }

  if (rlwe_ == nullptr) {
    // Without RLWE compression, we only support 2D constructions
    if (orig_dims_.size() != 2) {
      throw std::invalid_argument("Three or more dimensions are currently not "
                                  "supported without RLWE compression");
    }
  } else {
    if (rlwe_->cudastf_ctx_ != lwe_->cudastf_ctx_) {
      throw std::invalid_argument(
          "lwe_ and rlwe_ must use the same cudastf_ctx_");
    }
    if (rlwe_->N_ % tile_n_max_ != 0) {
      throw std::invalid_argument("rlwe_->N_ must be a multiple of " +
                                  std::to_string(tile_n_max_));
    }
  }

  // Check the degrees are valid
  if (lwe_->n_ % tile_m_max_ != 0) {
    throw std::invalid_argument("lwe_->n_ must be a multiple of " +
                                std::to_string(tile_m_max_));
  }

  // This context will be used for the subsequent dimension reduction steps
  if (orig_dims_.size() >= 3) {
    lwe_small_ = create_lwe_context<l_word>(lwe_->n_ - 1, lwe_->sigma_,
                                            lwe_->cudastf_ctx_);
  }

  for (size_t i = 0; i < orig_dims_.size(); i++) {
    if (orig_dims_.at(i) <= 1) {
      throw std::invalid_argument("each dimension must be greater than 1");
    }
  }

  prepare_first_reduction(std::move(db));
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::prepare_first_reduction(
    std::vector<plain_word> &&db) {
  if (orig_dims_.size() == 2) {
    prepare_first_reduction_2d(std::move(db));
    return;
  }

  // Currently, only three-dimensional constructions are supported
  size_t dim_first = orig_dims_.at(0);
  size_t dim_mid = orig_dims_.at(1);
  size_t dim_last = orig_dims_.at(2);

  // size_t dim_mid = std::accumulate(orig_dims_.begin() + 1,
  // orig_dims_.end() - 1,
  //                                 1, std::multiplies<size_t>());
  // size_t dim_last = orig_dims_.at(orig_dims_.size() - 1);
  if (db.size() != dim_first * dim_mid * dim_last) {
    throw std::invalid_argument(
        "database size must be equal to as defined by dims");
  }

  // Make sure that padded_dim_first is a multiple of tile_k_max_
  size_t padded_dim_first = pad_by<size_t>(dim_first, tile_k_max_);

  size_t num_gpus = device_ids_.size();
  size_t padded_dim_mid;
  // Make sure that the padded_dim_mid is a multiple of tile_k_max_ and
  // num_gpus
  {
    int tile_k_tmp = tile_k_max_;
    int num_gpus_tmp = num_gpus;
    int x, y;
    int gcd = extended_gcd(tile_k_tmp, num_gpus_tmp, x, y);
    size_t dim_mid_pad_granularity = tile_k_max_ * num_gpus / gcd;
    padded_dim_mid = pad_by<size_t>(dim_mid, dim_mid_pad_granularity);
  }
  size_t padded_dim_mid_per_gpu = padded_dim_mid / num_gpus;

  size_t padded_dim_last;
  // Make sure that (padded_dim_mid_per_gpu * dim_last) is a multiple of
  // tile_n_max_
  {
    int tile_n_tmp = tile_n_max_;
    int dim_mid_tmp = padded_dim_mid_per_gpu;
    int x, y;
    int gcd = extended_gcd(tile_n_tmp, dim_mid_tmp, x, y);
    size_t dim_last_pad_granularity = tile_n_max_ / gcd;
    padded_dim_last = pad_by<size_t>(dim_last, dim_last_pad_granularity);
  }

  padded_dims_.resize(3);
  padded_dims_.at(0) = padded_dim_first;
  padded_dims_.at(1) = padded_dim_mid;
  padded_dims_.at(2) = padded_dim_last;

  // Use CUDA APIs to copy the dim_first x dim_mid x dim_last tensor to
  // padded_dim_first x padded_dim_mid_per_gpu x padded_dim_last partitions
  // in each GPU

  size_t remaining_dim_mid = dim_mid;
  for (size_t i = 0; i < num_gpus; i++) {
    // Initialize the GPU memory
    plain_word *gpu_db_ptr;
    cudastf::cuda_safe_call(cudaSetDevice(device_ids_[i]));
    cudastf::cuda_safe_call(
        cudaMalloc(&gpu_db_ptr, padded_dim_first * padded_dim_mid_per_gpu *
                                    padded_dim_last * sizeof(plain_word)));
    cudastf::cuda_safe_call(
        cudaMemset(gpu_db_ptr, 0,
                   padded_dim_first * padded_dim_mid_per_gpu * padded_dim_last *
                       sizeof(plain_word)));

    device_db_ptrs_.push_back(gpu_db_ptr);
    size_t copy_dim_mid = std::min(remaining_dim_mid, padded_dim_mid_per_gpu);
    remaining_dim_mid -= copy_dim_mid;

    // Skip copying if the dimension is zero
    if (copy_dim_mid == 0)
      continue;

    // Otherwise, copy the data
    for (size_t j = 0; j < dim_last; j++) {
      plain_word *gpu_db_ptr_j =
          gpu_db_ptr + j * padded_dim_mid_per_gpu * padded_dim_first;
      plain_word *host_db_ptr_j = db.data() + j * dim_mid * dim_first +
                                  i * padded_dim_mid_per_gpu * dim_first;
      cudastf::cuda_safe_call(cudaMemcpy2D(
          gpu_db_ptr_j, padded_dim_first * sizeof(plain_word), host_db_ptr_j,
          dim_first * sizeof(plain_word), dim_first * sizeof(plain_word),
          copy_dim_mid, cudaMemcpyHostToDevice));
    }
  }
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::prepare_first_reduction_2d(
    std::vector<plain_word> &&db) {
  if (orig_dims_.size() != 2) {
    throw std::invalid_argument("dims must have exactly 2 elements");
  }
  size_t dim_first = orig_dims_.at(0);
  size_t dim_last = orig_dims_.at(1);
  if (db.size() != dim_first * dim_last) {
    throw std::invalid_argument(
        "database size must be equal to as defined by dims");
  }

  size_t num_gpus = device_ids_.size();
  size_t padded_dim_first = pad_by<size_t>(dim_first, tile_k_max_);
  size_t dim_last_granularity = tile_n_max_ * num_gpus;
  size_t padded_dim_last = pad_by<size_t>(dim_last, dim_last_granularity);
  size_t padded_dim_last_per_gpu = padded_dim_last / num_gpus;

  padded_dims_.resize(3);
  padded_dims_.at(0) = padded_dim_first;
  padded_dims_.at(1) = 1;
  padded_dims_.at(2) = padded_dim_last;

  size_t remaining_dim_last = dim_last;
  for (size_t i = 0; i < num_gpus; i++) {
    // Initialize the GPU memory
    plain_word *gpu_db_ptr;
    cudastf::cuda_safe_call(cudaSetDevice(device_ids_[i]));
    cudastf::cuda_safe_call(
        cudaMalloc(&gpu_db_ptr, padded_dim_first * padded_dim_last_per_gpu *
                                    sizeof(plain_word)));
    cudastf::cuda_safe_call(cudaMemset(
        gpu_db_ptr, 0,
        padded_dim_first * padded_dim_last_per_gpu * sizeof(plain_word)));

    device_db_ptrs_.push_back(gpu_db_ptr);

    size_t copy_dim_last =
        std::min(remaining_dim_last, padded_dim_last_per_gpu);
    remaining_dim_last -= copy_dim_last;

    // Skip copying if the dimension is zero
    if (copy_dim_last == 0)
      continue;

    // Otherwise, copy the data
    cudastf::cuda_safe_call(cudaMemcpy2D(
        gpu_db_ptr, padded_dim_first * sizeof(plain_word),
        db.data() + i * padded_dim_last_per_gpu * dim_first,
        dim_first * sizeof(plain_word), dim_first * sizeof(plain_word),
        copy_dim_last, cudaMemcpyHostToDevice));
  }
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::offline_setup() {
  size_t padded_dim_first = padded_dims_.at(0);
  size_t padded_dim_mid = padded_dims_.at(1); // can be 1
  size_t padded_dim_last = padded_dims_.at(2);

  auto &ctx = lwe_->cudastf_ctx_;

  // Prepare input query memory space
  cudastf::cuda_safe_call(cudaSetDevice(device_ids_.at(0)));
  l_word *query_ptr;
  size_t query_size = max_batch_size_ * padded_dim_first;
  cudastf::cuda_safe_call(cudaMalloc(&query_ptr, query_size * sizeof(l_word)));
  cudastf::cuda_safe_call(
      cudaMemset(query_ptr, 0, query_size * sizeof(l_word)));
  query_ld_.emplace_back(ctx->logical_data(
      query_ptr, query_size, cudastf::data_place::device(device_ids_.at(0))));

  // Prepare output response memory space
  cudastf::cuda_safe_call(cudaSetDevice(device_ids_.at(0)));

  std::vector<l_word> a_data;
  a_seed_.emplace_back();
  lwe_->generate_seed(a_seed_.at(0));
#ifdef DEBUG_ZERO_A
  a_data.resize(lwe_->n_ * padded_dim_first);
  std::fill(a_data.begin(), a_data.end(), 0);
#else
  lwe_->sample_a_from_seed(a_data, lwe_->n_ * padded_dim_first, a_seed_.at(0));
#endif
  auto a_ld = vector_to_ld(ctx, std::move(a_data));
  a_ld.set_symbol("SIMPLEPIR_A");

  std::vector<logical_data_t<cudastf::slice<l_word>>> multi_gpu_hint;
  db_gemm(multi_gpu_hint, a_ld, lwe_->n_);

  size_t num_gpus = device_ids_.size();

  // gather the hint (SimplePIR with hint)
  if (rlwe_ == nullptr) {
    if (padded_dim_mid != 1) {
      throw std::invalid_argument(
          "padded_dim_mid must be 1 when RLWE compression is not used");
    }
    hint_gathered_.resize(lwe_->n_ * padded_dim_last);

    size_t padded_dim_last_per_gpu = padded_dim_last / num_gpus;
    size_t per_gpu_hint_size = padded_dim_last_per_gpu * lwe_->n_;
    for (size_t i = 0; i < num_gpus; i++) {
      std::vector<l_word> tmp(per_gpu_hint_size);
      __copy_ld_to_host_ptr(ctx, tmp.data(), tmp.size(), multi_gpu_hint.at(i));
      for (size_t j = 0; j < lwe_->n_; j++) {
        std::copy(tmp.begin() + j * padded_dim_last_per_gpu,
                  tmp.begin() + (j + 1) * padded_dim_last_per_gpu,
                  hint_gathered_.begin() + i * padded_dim_last_per_gpu +
                      j * padded_dim_last);
      }
    }
    return;
  }

  // The offline phase includes performing NTT on a_data if using
  // RLWE compression
  size_t lwe_bits = sizeof(l_word) * 8 - lwe_round_shift_bits_;
  r_word rlwe_scale = rlwe_->modulus_ / (((r_word)1) << lwe_bits);
  size_t fldim, ntt_stride;
  if (padded_dim_mid == 1) {
    // 2D case (in this case, padded_dim_last is split into num_gpus
    fldim = padded_dim_last / num_gpus;
    ntt_stride = 1;
  } else {
    // 3D case (in this case, padded_dim_mid is split into num_gpus
    size_t padded_dim_mid_per_gpu = padded_dim_mid / num_gpus;
    fldim = padded_dim_mid_per_gpu * padded_dim_last;
    ntt_stride = padded_dim_mid_per_gpu;
  }

  // Create RLWE compressors for each GPU
  rlwe_compress_.emplace_back();
  for (size_t i = 0; i < num_gpus; i++) {
    RLWECompressPtr<l_word, r_word> compress =
        create_rlwe_compress<l_word, r_word>(lwe_, rlwe_, fldim, ntt_stride,
                                             rlwe_scale, rlwe_round_shift_bits_,
                                             device_ids_.at(i));
    compress->prepare_a_ntt(multi_gpu_hint.at(i));
    rlwe_compress_.at(0).push_back(compress);
  }

  // Prepare memory space for conversion ciphertexts
  cudastf::cuda_safe_call(cudaSetDevice(device_ids_.at(0)));
  r_word *conv_a_ptr, *conv_b_ptr;
  size_t conv_size = max_batch_size_ * lwe_->n_ * rlwe_->N_;
  cudastf::cuda_safe_call(cudaMalloc(&conv_a_ptr, conv_size * sizeof(r_word)));
  cudastf::cuda_safe_call(cudaMalloc(&conv_b_ptr, conv_size * sizeof(r_word)));
  cudastf::cuda_safe_call(
      cudaMemset(conv_a_ptr, 0, conv_size * sizeof(r_word)));
  cudastf::cuda_safe_call(
      cudaMemset(conv_b_ptr, 0, conv_size * sizeof(r_word)));
  conv_ctxt_ld_.emplace_back(ctx->logical_data(
      conv_a_ptr, conv_size, cudastf::data_place::device(device_ids_.at(0))));
  conv_ctxt_ld_.emplace_back(ctx->logical_data(
      conv_b_ptr, conv_size, cudastf::data_place::device(device_ids_.at(0))));

  if (orig_dims_.size() == 2) {
    return;
  }
  multi_gpu_hint.clear();

  // Prepare memory space for the seconnd query
  cudastf::cuda_safe_call(cudaSetDevice(device_ids_.at(0)));
  l_word *query_ptr_2;
  cudastf::cuda_safe_call(cudaMalloc(
      &query_ptr_2, max_batch_size_ * padded_dim_mid * sizeof(l_word)));
  cudastf::cuda_safe_call(cudaMemset(
      query_ptr_2, 0, max_batch_size_ * padded_dim_mid * sizeof(l_word)));
  query_ld_.emplace_back(
      ctx->logical_data(query_ptr_2, max_batch_size_ * padded_dim_mid,
                        cudastf::data_place::device(device_ids_.at(0))));

  // Prepare the second dimension hints
  std::vector<l_word> a_data_2;
  a_seed_.emplace_back();
  lwe_->generate_seed(a_seed_.at(1));
#ifdef DEBUG_ZERO_A
  a_data_2.resize(lwe_->n_ * padded_dim_mid);
  std::fill(a_data_2.begin(), a_data_2.end(), 0);
#else
  lwe_->sample_a_from_seed(a_data_2, (lwe_->n_ - 1) * padded_dim_mid,
                           a_seed_.at(1));
#endif
  a_data_2.resize(lwe_->n_ * padded_dim_mid);
  for (size_t i = 0; i < num_gpus; i++) {
    second_dim_a_.emplace_back(vector_to_ld(ctx, a_data_2));
    second_dim_a_.at(i).set_symbol("SIMPLEPIR_A_SECOND_DIM_GPU_" +
                                   std::to_string(i));
  }

  // Also prepare the next compression steps for 3D
  rlwe_compress_.emplace_back();
  fldim = pad_by<size_t>(padded_dim_last, rlwe_->N_) *
          get_num_rlwe_plain_word_chunks() * 2;
  ntt_stride = 1;

  for (size_t i = 0; i < num_gpus; i++) {
    RLWECompressPtr<l_word, r_word> compress =
        create_rlwe_compress<l_word, r_word>(lwe_, rlwe_, fldim, ntt_stride,
                                             rlwe_scale, rlwe_round_shift_bits_,
                                             device_ids_.at(i));
    rlwe_compress_.at(1).push_back(compress);
  }
}

template <typename l_word, typename r_word, typename plain_word>
std::vector<std::vector<plain_word>>
SimplePIRServer<l_word, r_word, plain_word>::online_stage_synchronized(
    std::vector<l_word *> &query, size_t batch_size,
    std::vector<r_word *> &conversion_ciphertexts) {
  size_t num_dims = orig_dims_.size();
  if (num_dims != query.size() + 1) {
    throw std::invalid_argument("query must have one less dimension than "
                                "orig_dims_");
  }
  if (batch_size > max_batch_size_) {
    throw std::invalid_argument("batch_size must be less than or equal to "
                                "max_batch_size_");
  }

  // Perform the first dimension redudction step
  auto &ctx = lwe_->cudastf_ctx_;
  size_t num_gpus = device_ids_.size();

  // First, create tasks to copy relevant data to the GPU
  // We just copy the data to the first GPU,
  // hopefully the other GPUS will take the data using fast GPU-to-GPU links
  if (query_ld_.size() != query.size()) {
    throw std::invalid_argument(
        "query_ld_.size() must be equal to query.size()");
  }
  for (size_t i = 0; i < query.size(); i++) {
    ctx->task(
        query_ld_.at(i).write(cudastf::data_place::device(device_ids_.at(0))))
            ->*[devid = device_ids_.at(0), query_pinned_ptr = query.at(i),
                size = batch_size * padded_dims_.at(i)](cudaStream_t stream,
                                                        auto query_ld_i) {
                  cudastf::cuda_safe_call(cudaSetDevice(devid));
                  cudastf::cuda_safe_call(cudaMemcpyAsync(
                      query_ld_i.data_handle(), query_pinned_ptr,
                      size * sizeof(l_word), cudaMemcpyHostToDevice, stream));
                };
  }
  if (conv_ctxt_ld_.size() != conversion_ciphertexts.size()) {
    throw std::invalid_argument(
        "conv_ctxt_ld_.size() must be equal to conversion_ciphertexts.size()");
  }
  for (size_t i = 0; i < conversion_ciphertexts.size(); i++) {
    ctx->task(conv_ctxt_ld_.at(i).write(
                  cudastf::data_place::device(device_ids_.at(0))))
            ->*[devid = device_ids_.at(0),
                conv_ctxt_pinned_ptr = conversion_ciphertexts.at(i),
                size = batch_size * lwe_->n_ * rlwe_->N_](cudaStream_t stream,
                                                          auto conv_ctxt_ld_i) {
                  cudastf::cuda_safe_call(cudaSetDevice(devid));
                  cudastf::cuda_safe_call(cudaMemcpyAsync(
                      conv_ctxt_ld_i.data_handle(), conv_ctxt_pinned_ptr,
                      size * sizeof(r_word), cudaMemcpyHostToDevice, stream));
                };
  }

  std::vector<logical_data_t<cudastf::slice<l_word>>> multi_gpu_res_b;
  db_gemm(multi_gpu_res_b, query_ld_.at(0),
          pad_by<size_t>(batch_size, tile_m_min_));

  std::vector<std::vector<plain_word>> res(batch_size);

  // 2D SimplePIR case
  if (rlwe_ == nullptr) {
    size_t eff_dim_last_per_gpu = padded_dims_.at(2) / num_gpus;
    size_t num_chunks = get_num_lwe_plain_word_chunks();
    for (size_t i = 0; i < num_gpus; i++) {
      std::vector<l_word> tmp(batch_size * eff_dim_last_per_gpu);
      __copy_ld_to_host_ptr(ctx, tmp.data(), tmp.size(), multi_gpu_res_b.at(i));

      // Do some post-processing for tmp
      for (size_t j = 0; j < batch_size; j++) {
        auto &dst_j = res.at(j);
        if (i == 0) {
          dst_j.resize(num_chunks * padded_dims_.at(2));
        }
        l_word *tmp_ptr_j = tmp.data() + j * eff_dim_last_per_gpu;
        for (size_t k = 0; k < eff_dim_last_per_gpu; k++) {
          l_word tmp_value = tmp_ptr_j[k];
          for (size_t l = 0; l < num_chunks; l++) {
            dst_j[l + (eff_dim_last_per_gpu * i + k) * num_chunks] =
                bit_truncate<plain_word, l_word>(tmp_value);
            tmp_value >>= (sizeof(plain_word) * 8);
          }
        }
      }
    }
    return res;
  }

  // auto conv_ctxt_ld = rlwe_->ciphertext_to_ld(conversion_ciphertexts);
  // auto &conv_ctxt_ld_a = conv_ctxt_ld.a_ld_;
  // auto &conv_ctxt_ld_b = conv_ctxt_ld.b_ld_;

  // Perform RLWE packing (first)
  std::vector<std::vector<logical_data_t<cudastf::slice<plain_word>>>>
      reinterpreted_res(num_gpus);
  for (size_t i = 0; i < num_gpus; i++) {
    auto &compress = rlwe_compress_.at(0).at(i);
    compress->multiply_conversion_ciphertexts_with_reinterpret(
        reinterpreted_res.at(i), conv_ctxt_ld_.at(0), conv_ctxt_ld_.at(1),
        multi_gpu_res_b.at(i), batch_size);
    if (reinterpreted_res.at(i).size() != batch_size) {
      throw std::invalid_argument(
          "reinterpreted_res.at(i).size() must be equal to batch_size");
    }
  }

  // In the 2D case, we just need to load the results back to host
  if (orig_dims_.size() == 2) {
    auto &compress = rlwe_compress_.at(0).at(0);
    size_t reinterpreted_size =
        compress->template get_reinterpreted_size<plain_word>();
    size_t copy_granularity = reinterpreted_size / 2;
    for (size_t i = 0; i < batch_size; i++) {
      auto &dst_i = res.at(i);
      dst_i.resize(reinterpreted_size * num_gpus);
      for (size_t j = 0; j < num_gpus; j++) {
        std::vector<plain_word> tmp(reinterpreted_size);
        __copy_ld_to_host_ptr(ctx, tmp.data(), tmp.size(),
                              reinterpreted_res.at(j).at(i));
        auto tmp_b_part = tmp.begin() + copy_granularity;
        std::copy(tmp.begin(), tmp_b_part,
                  dst_i.begin() + j * copy_granularity);
        std::copy(tmp_b_part, tmp.end(),
                  dst_i.begin() + (j + num_gpus) * copy_granularity);
      }
    }
    return res;
  }

  std::vector<logical_data_t<cudastf::slice<plain_word>>> *new_db_ptrs =
      &reinterpreted_res.at(0);
  std::vector<logical_data_t<cudastf::slice<plain_word>>> reordered_res;
  if (num_gpus > 1) {
    merge_data_across_gpus(reordered_res, reinterpreted_res);
    new_db_ptrs = &reordered_res;
  }

  // After merging, data is distributed across the GPUs in a batch-wise
  // manner.
  std::vector<logical_data_t<cudastf::slice<plain_word>>> final_res;
  second_dim_reduction(final_res, *new_db_ptrs, query_ld_.at(1));

  // Perform the second dimension reduction steps
  // In this case, we perform batch-wise processing
  if (final_res.size() != batch_size) {
    throw std::invalid_argument("final_res.size() must be equal to batch_size");
  }

  size_t reinterpreted_size =
      rlwe_compress_.at(1).at(0)->template get_reinterpreted_size<plain_word>();
  for (size_t i = 0; i < batch_size; i++) {
    auto &dst_i = res.at(i);
    dst_i.resize(reinterpreted_size);
    __copy_ld_to_host_ptr(ctx, dst_i.data(), dst_i.size(), final_res.at(i));
  }
  return res;
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::merge_data_across_gpus(
    std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
    std::vector<std::vector<logical_data_t<cudastf::slice<plain_word>>>>
        &reinterpreted_res) {
  res.clear();
  size_t num_gpus = device_ids_.size();
  if (num_gpus == 1) {
    std::cout << "WARNING: if num_gpus == 1, merge_data_across_gpus should not "
                 "be called"
              << std::endl;
    return;
  }

  if (reinterpreted_res.size() != num_gpus) {
    throw std::invalid_argument(
        "reinterpreted_res.size() must be equal to num_gpus");
  }
  size_t batch_size = reinterpreted_res.at(0).size();
  for (size_t i = 1; i < num_gpus; i++) {
    if (reinterpreted_res.at(i).size() != batch_size) {
      throw std::invalid_argument(
          "reinterpreted_res.at(i).size() must be equal to batch_size");
    }
  }

  size_t reinterpreted_size =
      rlwe_compress_.at(0).at(0)->template get_reinterpreted_size<plain_word>();
  size_t copy_granularity = reinterpreted_size / 2;
  for (size_t i = 0; i < batch_size; i++) {
    int devid = device_ids_.at(i % num_gpus);
    cudastf::cuda_safe_call(cudaSetDevice(devid));
    auto &ctx = lwe_->cudastf_ctx_;
    res.emplace_back(
        ctx->logical_data(cudastf::shape_of<cudastf::slice<plain_word>>(
                              reinterpreted_size * num_gpus))
            .set_symbol("Merged_Res_Batch_" + std::to_string(i)));
    for (size_t j = 0; j < num_gpus; j++) {
      ctx->cuda_kernel(res.at(i).write(cudastf::data_place::device(devid)),
                       reinterpreted_res.at(i).at(i).read(
                           cudastf::data_place::device(devid)))
              ->*
          [copy_granularity = copy_granularity, num_gpus = num_gpus,
           j = j](auto res_i, auto data_in) {
            plain_word *res_i_a = res_i.data_handle() + copy_granularity * j;
            plain_word *res_i_b =
                res_i.data_handle() + copy_granularity * (num_gpus + j);
            const plain_word *data_in_a = data_in.data_handle();
            const plain_word *data_in_b =
                data_in.data_handle() + copy_granularity;
            constexpr uint32_t num_threads = 256;
            uint32_t num_blocks =
                div_ceil<uint32_t>(copy_granularity, num_threads);
            if (num_blocks > 2048) {
              num_blocks = 2048;
            }

            return cudastf::cuda_kernel_desc{copy2_kernel<plain_word>,
                                             num_threads,
                                             num_blocks,
                                             0,
                                             res_i_a,
                                             res_i_b,
                                             data_in_a,
                                             data_in_b,
                                             copy_granularity};
          };
    }
  }
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::second_dim_reduction(
    std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
    std::vector<logical_data_t<cudastf::slice<plain_word>>> &new_db,
    logical_data_t<cudastf::slice<l_word>> &query) {
  // We use a small trick of merging (hint + query) to perform the second
  // dimension reduction step

  // To do so, we use lwe_small_, which uses a smaller (lwe_->n_ - 1) degree.
  // We merge the hint ((n - 1) x padded_dim_mid) and query (1 x
  // padded_dim_mid) into a (n x padded_dim_mid) matrix.

  // After multiplying it with the new_db, we get a
  // n x kN matrix, which can be split into n x N matrices.

  // Normally, we would split it into
  // res_a : (n - 1) x 1 x N tensor
  // res_b : 1 x 1 x N tensor

  // and for,
  // conv_a : 1 x (n - 1) x N tensor
  // conv_b : 1 x (n - 1) x N tensor

  // computes
  // final_res_a = conv_a * res_a (N-parallel GEMMs) --> 1 x 1 x N tensor
  // final_res_b = conv_b * res_a (N-parallel GEMMs) + res_b --> 1 x 1 x N
  // tensor

  // However, we can instead perform (notations are not very solid here)
  // final_res_a = [conv_a, 0] * [res_a, res_b]
  // final_res_b = [conv_b, 1] * [res_a, res_b]

  // Then, for RLWE packing, we apply NTT to the merged matrix.
  // Then for conversion ciphertexts: (conv_a, conv_b)
  // conv_a : n x N (matrix)
  // conv_b : n x N (matrix)

  size_t num_gpus = device_ids_.size();
  res.clear();

  size_t batch_size = new_db.size();

  for (size_t i = 0; i < batch_size; i++) {
    auto devid = device_ids_.at(i % num_gpus);
    cudastf::cuda_safe_call(cudaSetDevice(devid));
    auto &ctx = lwe_->cudastf_ctx_;

    // merge query with the hint
    auto &a_i = second_dim_a_.at(i % num_gpus);
    size_t eff_dim_rest = pad_by<size_t>(padded_dims_.at(2), rlwe_->N_) *
                          get_num_rlwe_plain_word_chunks() * 2;
    auto gemm_res = ctx->logical_data(cudastf::shape_of<cudastf::slice<l_word>>(
                                          lwe_->n_ * eff_dim_rest))
                        .set_symbol("Second_Dim_GEMM_Res_" + std::to_string(i));
    ctx->cuda_kernel_chain(
        gemm_res.write(cudastf::data_place::device(devid)),
        a_i.rw(cudastf::data_place::device(devid)),
        query.read(cudastf::data_place::device(devid)),
        new_db.at(i).read(cudastf::data_place::device(devid)))
            ->*
        [i = i, batch_size = batch_size, lwe_n = lwe_->n_,
         padded_dim_mid = padded_dims_.at(1), eff_dim_rest = eff_dim_rest,
         round_shift_bits = lwe_round_shift_bits_](auto gemm_res, auto a_i,
                                                   auto query, auto new_db_i) {
          std::vector<cudastf::cuda_kernel_desc> kernel_descs;

          if (a_i.size() != lwe_n * padded_dim_mid) {
            throw std::invalid_argument(
                "a_i.size() must be equal to lwe_n * padded_dim_mid");
          }
          if (query.size() < padded_dim_mid * batch_size) {
            throw std::invalid_argument("query.size() cannot be smaller than "
                                        "padded_dim_mid * batch_size");
          }
          if (new_db_i.size() != padded_dim_mid * eff_dim_rest) {
            throw std::invalid_argument("new_db_i.size() must be equal to "
                                        "padded_dim_mid * eff_dim_rest");
          }
          if (gemm_res.size() != lwe_n * eff_dim_rest) {
            throw std::invalid_argument(
                "gemm_res.size() must be equal to lwe_n * eff_dim_rest");
          }

          // Copy query to the hint
          constexpr uint32_t num_threads = 256;
          uint32_t num_blocks = div_ceil<uint32_t>(padded_dim_mid, num_threads);
          if (num_blocks > 2048) {
            num_blocks = 2048;
          }
          auto copy_kernel = cudastf::cuda_kernel_desc{
              copy1_kernel<l_word>,
              num_blocks,
              num_threads,
              0,
              a_i.data_handle() + padded_dim_mid * (lwe_n - 1),
              query.data_handle() + padded_dim_mid * i,
              padded_dim_mid};
          kernel_descs.push_back(copy_kernel);

          // Perform GEMM
          auto gemm_kernel = get_gemm_kernel_desc(
              lwe_n, eff_dim_rest, padded_dim_mid, a_i.data_handle(),
              new_db_i.data_handle(), gemm_res.data_handle(), round_shift_bits);
          kernel_descs.push_back(gemm_kernel);

          return kernel_descs;
        };

    std::vector<logical_data_t<cudastf::slice<plain_word>>> tmp_res;
    rlwe_compress_.at(1)
        .at(i % num_gpus)
        ->do_everything_per_batch(tmp_res, gemm_res, conv_ctxt_ld_.at(0),
                                  conv_ctxt_ld_.at(1), i);
    if (tmp_res.size() != 1) {
      throw std::invalid_argument("tmp_res.size() must be equal to 1");
    }

    res.emplace_back(std::move(tmp_res.at(0)));
  }
}

template <typename l_word, typename r_word, typename plain_word>
void SimplePIRServer<l_word, r_word, plain_word>::db_gemm(
    std::vector<logical_data_t<cudastf::slice<l_word>>> &multi_gpu_res,
    logical_data_t<cudastf::slice<l_word>> &data_in, size_t M) {
  // data_in is a M x K matrix
  // db (per_gpu) is a N x K matrix
  // res (per_gpu) = data_in * db^T is a M x N matrix
  // M == batch_size
  // K == dim_first_
  // N == eff_dim_rest_
  static_assert(std::is_same_v<l_word, int32_t>,
                "Currently, word must be int32_t");
  static_assert(std::is_same_v<plain_word, int8_t>,
                "Currently, plain_word must be int8_t");

  multi_gpu_res.clear();
  size_t num_gpus = device_ids_.size();
  size_t eff_dim_rest_per_gpu =
      padded_dims_.at(1) * padded_dims_.at(2) / num_gpus;

  // M must be a multiple of tile_m
  size_t tile_m = get_gemm_m_tile_size(M);
  if (M % tile_m != 0) {
    throw std::invalid_argument(
        "M must be a multiple of tile_m (tile_m depends on M)");
  }

  for (size_t i = 0; i < num_gpus; i++) {
    int devid = device_ids_[i];
    cudastf::cuda_safe_call(cudaSetDevice(devid));
    auto &ctx = lwe_->cudastf_ctx_;
    auto res_i = ctx->logical_data(cudastf::shape_of<cudastf::slice<l_word>>(
                                       M * eff_dim_rest_per_gpu))
                     .set_symbol("DB_GEMM_Res_GPU" + std::to_string(devid));
    ctx->cuda_kernel(res_i.write(cudastf::data_place::device(devid)),
                     data_in.read(cudastf::data_place::device(devid)))
            .set_symbol("DB_GEMM_GPU" + std::to_string(devid))
            ->*
        [M = M, N = eff_dim_rest_per_gpu, K = padded_dims_.at(0),
         db_i = device_db_ptrs_[i],
         round_shift_bits = lwe_round_shift_bits_](auto res_i, auto data_in) {
          if (M * K != data_in.size()) {
            throw std::invalid_argument(
                "data_in.size() must be equal to M * K");
          }
          return get_gemm_kernel_desc(M, N, K, data_in.data_handle(), db_i,
                                      res_i.data_handle(), round_shift_bits);
        };

    multi_gpu_res.emplace_back(std::move(res_i));
  }
}

template class SimplePIRServer<int32_t, uint64_t, int8_t>;

} // namespace lwe_ann