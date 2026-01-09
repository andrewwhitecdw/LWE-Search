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
 
#include "../rlwe_kernels/constants.h"
#include "../rlwe_kernels/ntt_rlwe.cuh"
#include "arith_utils.h"
#include "rlwe_compress.h"
#include "type_utils.h"

#include <algorithm>

#define FFT_FORWARD -1
#define FFT_INVERSE 1

#define GEMM_WORKER_BODY                                                       \
  dim3 block_dim(th_x *th_y *th_z);                                            \
  dim3 grid_dim(ntt_stride / tile_ntt_stride,                                  \
                div_ceil<size_t>(batch_size, tile_batch_size),                 \
                N / copy_granularity_r_words);                                 \
  unsigned int smem_size =                                                     \
      (tile_ntt_stride + tile_batch_size) * tile_n * copy_granularity_bytes;   \
  return cudastf::cuda_kernel_desc{                                            \
      gemm_for_conversion<r_word, tile_ntt_stride, tile_batch_size, tile_n,    \
                          th_x, th_y, th_z, accumulate,                        \
                          second_dim_overwrite_ct>,                            \
      grid_dim,                                                                \
      block_dim,                                                               \
      smem_size,                                                               \
      res.data_handle(),                                                       \
      a_ntt_res.data_handle(),                                                 \
      conversion_data_ptr,                                                     \
      N,                                                                       \
      n,                                                                       \
      ntt_stride,                                                              \
      batch_size,                                                              \
      modulus,                                                                 \
      barrett_k,                                                               \
      barrett_rcp};

namespace lwe_ann::detail {

__device__ inline void copy_128b(void *dst, const void *src) {
  *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
}

template <unsigned int bytes> struct bytes_helper {
  static_assert(is_pow2(bytes), "bytes must be a power of 2");
  static_assert(bytes <= 16, "bytes must be less than or equal to 16");

  using type = std::conditional_t<
      bytes == 1, uint8_t,
      std::conditional_t<
          bytes == 2, uint16_t,
          std::conditional_t<bytes == 4, uint32_t,
                             std::conditional_t<bytes == 8, uint64_t, int4>>>>;
};

template <unsigned int bytes>
using bytes_t = typename bytes_helper<bytes>::type;

template <unsigned int bytes>
__device__ inline void copy_bytes(void *dst, const void *src) {
  if constexpr (bytes >= 16) {
    *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    copy_bytes<bytes - 16>(reinterpret_cast<uint8_t *>(dst) + 16,
                           reinterpret_cast<const uint8_t *>(src) + 16);
  } else if constexpr (bytes >= 8) {
    *reinterpret_cast<uint64_t *>(dst) =
        *reinterpret_cast<const uint64_t *>(src);
    copy_bytes<bytes - 8>(reinterpret_cast<uint8_t *>(dst) + 8,
                          reinterpret_cast<const uint8_t *>(src) + 8);
  } else if constexpr (bytes >= 4) {
    *reinterpret_cast<uint32_t *>(dst) =
        *reinterpret_cast<const uint32_t *>(src);
    copy_bytes<bytes - 4>(reinterpret_cast<uint8_t *>(dst) + 4,
                          reinterpret_cast<const uint8_t *>(src) + 4);
  } else if constexpr (bytes >= 2) {
    *reinterpret_cast<uint16_t *>(dst) =
        *reinterpret_cast<const uint16_t *>(src);
    copy_bytes<bytes - 2>(reinterpret_cast<uint8_t *>(dst) + 2,
                          reinterpret_cast<const uint8_t *>(src) + 2);
  } else if constexpr (bytes == 1) {
    *reinterpret_cast<uint8_t *>(dst) = *reinterpret_cast<const uint8_t *>(src);
  }
  // Do nothing for bytes == 0
}

template <typename l_word, typename r_word, unsigned int load_words,
          unsigned int tile_N, unsigned int tile_ntt_stride, bool merged>
__global__ void copy_for_ntt(r_word *ntt_res, const l_word *lwe_data,
                             make_signed_t<r_word> rlwe_scale, r_word modulus,
                             unsigned int N, unsigned int fldim,
                             unsigned int ntt_stride,
                             unsigned int current_ntt_column_size) {
  using sr_word = make_signed_t<r_word>;
  // lwe_data: batch_size x (N x ntt_stride) chunk of batch_size x fldim matrix
  // ntt_res: batch_size x (ntt_stride x N) matrix

  static_assert(is_pow2(load_words), "load_words must be a power of 2");
  static_assert(sizeof(l_word) * load_words <= 16,
                "load_words * sizeof(l_word) must be less than or equal to 16");
  constexpr unsigned int load_bytes = load_words * sizeof(l_word);
  using load_copy_t = bytes_t<load_bytes>;

  static_assert(tile_ntt_stride % load_words == 0,
                "tile_ntt_stride must be divisible by load_words");
  constexpr unsigned int load_group = tile_ntt_stride / load_words;
  static_assert(is_pow2(tile_N), "tile_N must be a power of 2");

  unsigned int ntt_stride_idx =
      (threadIdx.x % load_group) * load_words + blockIdx.x * tile_ntt_stride;
  unsigned int N_idx = threadIdx.x / load_group + blockIdx.y * tile_N;
  unsigned int batch_idx = blockIdx.z;

  sr_word tmp_scaled[load_words] = {0}; // Zero-initialize
  if (N_idx < current_ntt_column_size) {
    load_copy_t tmp;
    copy_bytes<load_bytes>(&tmp,
                           &lwe_data[batch_idx * static_cast<size_t>(fldim) +
                                     N_idx * ntt_stride + ntt_stride_idx]);
#pragma unroll
    for (unsigned int i = 0; i < load_words; i++) {
      if constexpr (merged) {
        if (batch_idx == gridDim.z - 1) {
          tmp_scaled[i] =
              static_cast<sr_word>(reinterpret_cast<l_word *>(&tmp)[i]) *
              rlwe_scale;
        } else {
          tmp_scaled[i] =
              static_cast<sr_word>(reinterpret_cast<l_word *>(&tmp)[i]);
        }
      } else {
        tmp_scaled[i] =
            static_cast<sr_word>(reinterpret_cast<l_word *>(&tmp)[i]) *
            rlwe_scale;
      }
      if (tmp_scaled[i] < 0) {
        tmp_scaled[i] += modulus;
      }
    }
  }

  // Store the results to ntt_res (this will likely result in coalesced access
  // for small load_group)
#pragma unroll
  for (unsigned int i = 0; i < load_words; i++) {
    ntt_res[batch_idx * static_cast<size_t>(ntt_stride) * N +
            (ntt_stride_idx + i) * N + N_idx] =
        static_cast<r_word>(tmp_scaled[i]);
  }
}

// This works when ntt_stride >= tile_ntt_stride (e.g., >= 64)
template <typename r_word, unsigned int tile_ntt_stride,
          unsigned int tile_batch_size, unsigned int tile_n, unsigned int th_x,
          unsigned int th_y, unsigned int th_z, bool accumulate = false,
          int second_dim_overwrite_ct = -1>
__global__ void
gemm_for_conversion(r_word *res, const r_word *a_data, const r_word *ct_data,
                    unsigned int N, unsigned int n, unsigned int ntt_stride,
                    unsigned int batch_size, r_word modulus,
                    unsigned int barrett_k, r_word barrett_rcp) {
  using sr_word = lwe_ann::make_signed_t<r_word>;
  using dr_word = lwe_ann::make_dword_t<r_word>;

  static_assert(
      second_dim_overwrite_ct == -1 || !accumulate,
      "when second_dim_overwrite_ct is not -1, accumulate must be false");
  static_assert(second_dim_overwrite_ct >= -1 && second_dim_overwrite_ct <= 1,
                "second_dim_overwrite_ct can only be -1, 0, or 1");

  // Coalesced access granularity
  constexpr unsigned int copy_granularity_bytes = 32;
  // We only need to care about the granularity of the r_word
  constexpr unsigned int r_words_per_128b = 16 / sizeof(r_word);   // -> 2
  constexpr unsigned int copy_group = copy_granularity_bytes / 16; // -> 2
  constexpr unsigned int copy_granularity_r_words =
      copy_granularity_bytes / sizeof(r_word); // -> 4

  constexpr int4 zero_128b = {0, 0, 0, 0};

  extern __shared__ int4 __smem[];
  r_word *sa = reinterpret_cast<r_word *>(__smem);
  r_word *sb = sa + tile_ntt_stride * tile_n * copy_granularity_r_words;

  // Copy a_data to shared memory

  // a_data: n x ntt_stride x N (row-major)
  // ct_data: batch_size x n x N (row-major)
  // Perform N-parallel GEMMs to produce
  // res: batch_size x ntt_stride x N (row-major)

  // blockIdx.x : 0 ~ ntt_stride / tile_ntt_stride - 1
  // blockIdx.y : 0 ~ batch_size / tile_batch_size - 1
  // blockIdx.z : 0 ~ N / copy_granularity_r_words - 1

  const r_word *a_data_ptr = a_data + copy_granularity_r_words * blockIdx.z;
  a_data_ptr += N * static_cast<size_t>(tile_ntt_stride) * blockIdx.x;
  const r_word *ct_data_ptr = ct_data + copy_granularity_r_words * blockIdx.z;
  // ct_data_ptr += N * n * static_cast<size_t>(tile_batch_size) * blockIdx.y;
  r_word *res_ptr = res + copy_granularity_r_words * blockIdx.z;
  res_ptr += N * static_cast<size_t>(tile_ntt_stride) * blockIdx.x;
  // res_ptr += N * ntt_stride * static_cast<size_t>(tile_batch_size) *
  // blockIdx.y;

  // thread_dim_3d.x determines how the threads will be organized
  // x-axis: partitioning tile_ntt_stride
  // y-axis: partitioning tile_batch_size
  // z-axis: copy_group
  // constexpr dim3 thread_dim_3d(16, num_threads / 16 / copy_group,
  // copy_group);
  // -> (16, 8, 2)
  // constexpr unsigned int num_threads = th_x * th_y * th_z;
  // static_assert(num_threads ==
  //                   thread_dim_3d.x * thread_dim_3d.y * thread_dim_3d.z,
  //               "num_threads must be divisible by (copy_group * 16)");
  static_assert(tile_ntt_stride % th_x == 0,
                "tile_ntt_stride must be divisible by th_x");
  static_assert(tile_batch_size % th_y == 0,
                "tile_batch_size must be divisible by th_y");
  static_assert(th_z == copy_group, "th_z must be equal to copy_group");
  // static_assert(N % thread_dim_3d.z == 0,
  //               "N must be divisible by thread_dim_3d.z");

  constexpr unsigned int ntt_stride_iters = tile_ntt_stride / th_x;
  constexpr unsigned int batch_size_iters = tile_batch_size / th_y;

  // Remapping threadIdx.x to 3D coordinates
  // Order: z -> y -> x (reversed)
  auto tid_3d_xy = threadIdx.x / th_z;
  dim3 tid_3d(tid_3d_xy / th_y, tid_3d_xy % th_y, threadIdx.x % th_z);

  // Add copy-granularity offset
  a_data_ptr += tid_3d.z * r_words_per_128b;
  ct_data_ptr += tid_3d.z * r_words_per_128b;
  res_ptr += tid_3d.z * r_words_per_128b;
  sa += tid_3d.z * tile_ntt_stride * tile_n * r_words_per_128b;
  sb += tid_3d.z * tile_batch_size * tile_n * r_words_per_128b;

  // Initialize the result registers
  int4 res_reg[batch_size_iters][ntt_stride_iters];
  for (unsigned int i = 0; i < batch_size_iters; i++) {
    unsigned int global_batch_idx =
        tile_batch_size * blockIdx.y + i * th_y + tid_3d.y;
    for (unsigned int j = 0; j < ntt_stride_iters; j++) {
      if constexpr (accumulate) {
        if (global_batch_idx >= batch_size) {
          res_reg[i][j] = zero_128b;
          continue;
        }
        unsigned int ntt_stride_idx = j * th_x + tid_3d.x;
        copy_128b(&res_reg[i][j],
                  res_ptr +
                      global_batch_idx * static_cast<size_t>(ntt_stride) * N +
                      ntt_stride_idx * N);
      } else {
        res_reg[i][j] = zero_128b;
      }
    }
  }

  // Main loop
  for (unsigned int n_iter = 0; n_iter < (n / tile_n); n_iter++) {
    // Synchronize before copying
    if (n_iter > 0) {
      __syncthreads();
    }

    // Copy a_data to shared memory
    for (unsigned int i = tid_3d_xy; i < tile_ntt_stride * tile_n;
         i += th_x * th_y) {
      unsigned int ntt_stride_idx = i % tile_ntt_stride;
      unsigned int n_idx = i / tile_ntt_stride;

      int4 copy_tmp;
      copy_128b(&copy_tmp, a_data_ptr +
                               n_idx * static_cast<size_t>(ntt_stride) * N +
                               ntt_stride_idx * N);
#pragma unroll
      for (unsigned int j = 0; j < r_words_per_128b; j++) {
        sa[j * tile_ntt_stride * tile_n + i] =
            reinterpret_cast<r_word *>(&copy_tmp)[j];
      }
    }

    // Copy ct_data to shared memory
    for (unsigned int i = tid_3d_xy; i < tile_batch_size * tile_n;
         i += th_x * th_y) {
      unsigned int batch_idx = i % tile_batch_size;
      unsigned int n_idx = i / tile_batch_size;
      unsigned int global_batch_idx = tile_batch_size * blockIdx.y + batch_idx;
      int4 copy_tmp;
      if (global_batch_idx >= batch_size) {
        copy_tmp = zero_128b;
      } else {
        copy_128b(&copy_tmp, ct_data_ptr + n_idx * N +
                                 global_batch_idx * static_cast<size_t>(n) * N);
        if constexpr (second_dim_overwrite_ct != -1) {
          if ((n_iter == (n / tile_n) - 1) && (n_idx == tile_n - 1)) {
            for (unsigned int j = 0; j < r_words_per_128b; j++) {
              reinterpret_cast<r_word *>(&copy_tmp)[j] =
                  static_cast<r_word>(second_dim_overwrite_ct);
            }
          }
        }
      }

#pragma unroll
      for (unsigned int j = 0; j < r_words_per_128b; j++) {
        sb[j * tile_batch_size * tile_n + i] =
            reinterpret_cast<r_word *>(&copy_tmp)[j];
      }
    }

    // Synchronize before gemm
    __syncthreads();

    // Perform GEMM
    for (unsigned int i = 0; i < batch_size_iters; i++) {
      unsigned int batch_idx = i * th_y + tid_3d.y;
      for (unsigned int j = 0; j < ntt_stride_iters; j++) {
        unsigned int ntt_stride_idx = j * th_x + tid_3d.x;
        for (unsigned int k = 0; k < r_words_per_128b; k++) {
          dr_word acc = reinterpret_cast<r_word *>(&res_reg[i][j])[k];
          r_word *sa_ptr = sa + ntt_stride_idx + k * tile_ntt_stride * tile_n;
          r_word *sb_ptr = sb + batch_idx + k * tile_batch_size * tile_n;
          for (unsigned int l = 0; l < tile_n; l++) {
            acc += static_cast<dr_word>(sa_ptr[l * tile_ntt_stride]) *
                   sb_ptr[l * tile_batch_size];
          }
          r_word acc_low = static_cast<r_word>(acc);
          r_word acc_high = static_cast<r_word>(acc >> (sizeof(r_word) * 8));
          dr_word tmp_low = static_cast<dr_word>(acc_low) * barrett_rcp;
          dr_word tmp_high = static_cast<dr_word>(acc_high) * barrett_rcp;
          dr_word tmp_shift = (tmp_low >> barrett_k) +
                              (tmp_high >> (barrett_k - sizeof(r_word) * 8));
          r_word res = static_cast<r_word>(acc - tmp_shift * modulus);
          if (res >= modulus) {
            res -= modulus;
          }
          reinterpret_cast<r_word *>(&res_reg[i][j])[k] = res;
        }
      }
    }
    a_data_ptr += tile_n * static_cast<size_t>(ntt_stride) * N;
    ct_data_ptr += tile_n * N;
  }

  // Copy back to global memory
  for (unsigned int i = 0; i < batch_size_iters; i++) {
    unsigned int global_batch_idx =
        tile_batch_size * blockIdx.y + i * th_y + tid_3d.y;
    if (global_batch_idx >= batch_size) {
      return;
    }
    for (unsigned int j = 0; j < ntt_stride_iters; j++) {
      unsigned int ntt_stride_idx = j * th_x + tid_3d.x;
      copy_128b(res_ptr +
                    global_batch_idx * static_cast<size_t>(ntt_stride) * N +
                    ntt_stride_idx * N,
                &res_reg[i][j]);
    }
  }
}

template <typename r_word, typename r_short>
__global__ void
rlwe_round_results(cudastf::slice<r_short> res, cudastf::slice<r_word> long_res,
                   r_word modulus, unsigned int round_shift_bits) {
  r_word round_constant = 0;
  if (round_shift_bits > 0) {
    round_constant = ((r_word)1) << (round_shift_bits - 1);
  }
  STRIDED_LOOP_START(res.size(), i);
  r_word tmp = long_res[i];
  tmp += round_constant;
  if (tmp >= modulus) {
    tmp -= modulus;
  }
  tmp >>= round_shift_bits;
  res[i] = static_cast<r_short>(tmp);
  STRIDED_LOOP_END;
}

template <typename r_word, typename plain_word, unsigned int load_words,
          unsigned int tile_N, unsigned int tile_ntt_stride>
__global__ void rlwe_reinterpret(plain_word *res, const r_word *long_res,
                                 unsigned int N, unsigned int ntt_stride,
                                 r_word modulus, unsigned int round_shift_bits,
                                 unsigned int num_plain_word_chunks) {
  using up_word = make_unsigned_t<plain_word>;

  // size_t pnnc64 = padded_num_ntt_columns;
  // res: ntt_stride x num_plain_word_chunks x pnnc x 2 (a or b)
  // long_res: N x ntt_stride

  static_assert(is_pow2(load_words), "load_words must be a power of 2");
  static_assert(
      sizeof(plain_word) * load_words <= 16,
      "load_words * sizeof(plain_word) must be less than or equal to 16");
  constexpr unsigned int load_bytes = load_words * sizeof(plain_word);
  using store_copy_t = bytes_t<load_bytes>;

  static_assert(tile_ntt_stride % load_words == 0,
                "tile_ntt_stride must be divisible by load_words");
  constexpr unsigned int load_group = tile_ntt_stride / load_words;
  static_assert(is_pow2(tile_N), "tile_N must be a power of 2");

  unsigned int ntt_stride_idx =
      (threadIdx.x % load_group) * load_words + blockIdx.x * tile_ntt_stride;
  unsigned int N_idx = threadIdx.x / load_group + blockIdx.y * tile_N;

  // Load and round the results
  r_word round_constant =
      round_shift_bits > 0 ? ((r_word)1) << (round_shift_bits - 1) : 0;
  r_word tmp[load_words];
#pragma unroll
  for (unsigned int i = 0; i < load_words; i++) {
    tmp[i] = long_res[N_idx + (ntt_stride_idx + i) * N];
  }
#pragma unroll
  for (unsigned int i = 0; i < load_words; i++) {
    tmp[i] += round_constant;
    if (tmp[i] >= modulus) {
      tmp[i] -= modulus;
    }
    tmp[i] >>= round_shift_bits;
  }

  // Store the results to res
#pragma unroll
  for (unsigned int i = 0; i < num_plain_word_chunks; i++) {
    store_copy_t tmp_store = {0};
    up_word *tmp_store_ptr = reinterpret_cast<up_word *>(&tmp_store);
    for (unsigned int j = 0; j < load_words; j++) {
      tmp_store_ptr[j] =
          static_cast<up_word>(tmp[j] >> (sizeof(plain_word) * 8 * i));
    }
    copy_bytes<load_bytes>(
        &res[(N_idx * num_plain_word_chunks + i) * ntt_stride + ntt_stride_idx],
        &tmp_store);
  }
}

template <typename r_word, typename plain_word, typename plain_word_save_t>
__global__ void rlwe_reinterpret_legacy(plain_word *res, const r_word *long_res,
                                        unsigned int N, unsigned int ntt_stride,
                                        unsigned int res_size_per_batch,
                                        r_word modulus,
                                        unsigned int round_shift_bits,
                                        unsigned int num_plain_word_chunks) {
  // Coalesced access granularity
  constexpr unsigned int copy_granularity_bytes = 32;
  // We only need to care about the granularity of the r_word
  constexpr unsigned int r_words_per_128b = 16 / sizeof(r_word);   // -> 2
  constexpr unsigned int copy_group = copy_granularity_bytes / 16; // -> 2
  constexpr unsigned int copy_granularity_r_words =
      copy_granularity_bytes / sizeof(r_word); // -> 4

  constexpr unsigned int ntt_stride_per_iter =
      sizeof(plain_word_save_t) / sizeof(plain_word);
  static_assert(64 % ntt_stride_per_iter == 0,
                "ntt_stride_per_iter must be a divisor of 64");

  r_word round_constant =
      round_shift_bits > 0 ? ((r_word)1) << (round_shift_bits - 1) : 0;

  // Apply N and batch_dix offsets
  unsigned int N_idx = blockIdx.x * copy_granularity_r_words +
                       (threadIdx.x % copy_group) * r_words_per_128b;

  long_res += (N_idx + blockIdx.y * ntt_stride * N);
  res += (N_idx * num_plain_word_chunks * ntt_stride +
          blockIdx.y * res_size_per_batch);

  unsigned int ntt_stride_idx_incr =
      (blockDim.x / copy_group) * ntt_stride_per_iter;

  for (unsigned int ntt_stride_idx =
           threadIdx.x / copy_group * ntt_stride_per_iter;
       ntt_stride_idx < ntt_stride; ntt_stride_idx += ntt_stride_idx_incr) {
    // Given that ntt_stride_per_iter is a divisor of 64
    // (ntt_stride is guaranteed to be a multiple of 64)
    // We can ignore out of bounds issues here.
    int4 res_reg[ntt_stride_per_iter];
#pragma unroll
    for (unsigned int i = 0; i < ntt_stride_per_iter; i++) {
      res_reg[i] =
          reinterpret_cast<const int4 *>(res + (ntt_stride_idx + i) * N)[0];
    }

// round the results
#pragma unroll
    for (unsigned int i = 0; i < ntt_stride_per_iter; i++) {
      for (unsigned int j = 0; j < r_words_per_128b; j++) {
        r_word tmp = reinterpret_cast<r_word *>(&res_reg[i])[j];
        tmp += round_constant;
        if (tmp >= modulus) {
          tmp -= modulus;
        }
        tmp >>= round_shift_bits;
        reinterpret_cast<r_word *>(&res_reg[i])[j] = tmp;
      }
    }

    // Store them in a ntt_stride_per_iter-granularity.
    constexpr r_word bit_mask = (((r_word)1) << (sizeof(plain_word) * 8)) - 1;
#pragma unroll
    for (unsigned int i = 0; i < r_words_per_128b; i++) {
      for (unsigned int j = 0; j < num_plain_word_chunks; j++) {
        plain_word_save_t tmp = 0;
        for (int k = ntt_stride_per_iter - 1; k >= 0; k--) {
          r_word tmp_r_word = reinterpret_cast<r_word *>(&res_reg[k])[i];
          tmp_r_word >>= (sizeof(plain_word) * 8 * j);
          tmp <<= (sizeof(plain_word) * 8);
          tmp |= (tmp_r_word & bit_mask);
        }
        *reinterpret_cast<plain_word_save_t *>(res + ntt_stride_idx +
                                               (i * num_plain_word_chunks + j) *
                                                   ntt_stride) = tmp;
      }
    }
  }
}

template <typename l_word, typename r_word>
void RLWECompress<l_word, r_word>::prepare_a_ntt(
    logical_data_t<cudastf::slice<l_word>> &a_data) {
  // We want to apply the compression along the final dimension of the
  // database.
  a_ntt_results_.clear();

  auto cudastf_ctx = lwe_->cudastf_ctx_;

  // fldim is a flattened dimension
  // Size of a_data: n x fldim (row-major)
  // fldim = ntt_column_size x ntt_stride

  // We first want to perform NTT on a_data
  // Regard each row of a_data as a two-dimensional structure of (fldim /
  // ntt_stride) x ntt_stride We need to perform NTT along the column
  // direction.

  // However, it is best to have ntt along contiguous data elements.
  // Therefore, we copy n x N x 128 chunk of a_data and reorganize
  // it as n x 128 x N and perform NTT along the last dimension.
  // We need to repeat this for (ntt_column_size / N) * (ntt_stride / 128)
  // times.

  for (size_t i = 0, ntt_column_index = 0; ntt_column_index < ntt_column_size_;
       i++, ntt_column_index += rlwe_->N_) {
    size_t current_ntt_column_size =
        std::min(ntt_column_size_ - ntt_column_index, rlwe_->N_);
    a_ntt_results_.emplace_back(
        cudastf_ctx
            ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                rlwe_->N_ * lwe_->n_ * ntt_stride_))
            .set_symbol("a_ntt_buffer_" + std::to_string(i)));
    auto &ntt_res = a_ntt_results_.back();

    copy_and_ntt<false>(ntt_res, a_data, ntt_column_index,
                        current_ntt_column_size, lwe_->n_, false);
  }
}

template <typename l_word, typename r_word>
template <bool merged>
void RLWECompress<l_word, r_word>::copy_and_ntt(
    logical_data_t<cudastf::slice<r_word>> &ntt_res,
    logical_data_t<cudastf::slice<l_word>> &lwe_data, size_t ntt_column_index,
    size_t current_ntt_column_size, size_t batch_size, bool is_b_data) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;
  sr_word rlwe_scale = (is_b_data || merged) ? rlwe_scale_ : 1;

  if (rlwe_->N_ % 256 != 0) {
    throw std::invalid_argument(
        "RLWECompress::copy_and_ntt: rlwe_->N_ must be divisible by 256");
  }

  cudastf::cuda_safe_call(cudaSetDevice(device_id_));
  cudastf_ctx
          ->cuda_kernel(ntt_res.write(cudastf::data_place::device(device_id_)),
                        lwe_data.read(cudastf::data_place::device(device_id_)))
          .set_symbol("copy_for_ntt")
          ->*
      [N = rlwe_->N_, batch_size = batch_size, fldim = this->fldim_,
       ntt_stride = this->ntt_stride_, ntt_column_index = ntt_column_index,
       current_ntt_column_size = current_ntt_column_size,
       rlwe_scale = rlwe_scale,
       modulus = rlwe_->modulus_](auto ntt_res, auto lwe_data) {
        // Check input sizes
        // if (lwe_data.size() != batch_size * fldim) {
        //   throw std::invalid_argument(
        //       "RLWECompress::copy_for_ntt: lwe_data.size() must be equal "
        //       "to batch_size * fldim");
        // }
        if (ntt_res.size() != N * batch_size * ntt_stride) {
          throw std::invalid_argument(
              "RLWECompress::copy_for_ntt: ntt_res.size() must be equal "
              "to N * batch_size * ntt_stride");
        }
        auto ntt_res_ptr = ntt_res.data_handle();
        auto lwe_data_ptr =
            lwe_data.data_handle() + ntt_column_index * ntt_stride;

        // Default
        constexpr dim3 block_dim(256);

        constexpr uint32_t target_load_words = 16 / sizeof(l_word);
        constexpr uint32_t target_tile_ntt_stride = 32 / sizeof(l_word);

        if (ntt_stride % 2 != 0) {
          constexpr uint32_t load_words = 1;
          constexpr uint32_t tile_ntt_stride = 1;
          constexpr uint32_t tile_N =
              block_dim.x / (tile_ntt_stride / load_words);

          dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N, batch_size);
          return cudastf::cuda_kernel_desc{
              copy_for_ntt<l_word, r_word, load_words, tile_N, tile_ntt_stride,
                           merged>,
              grid_dim,
              block_dim,
              0,
              ntt_res_ptr,
              lwe_data_ptr,
              rlwe_scale,
              modulus,
              N,
              fldim,
              ntt_stride,
              current_ntt_column_size};
        } else if (ntt_stride % 4 != 0) {
          constexpr uint32_t load_words =
              std::min<uint32_t>(target_load_words, 2);
          constexpr uint32_t tile_ntt_stride =
              std::min<uint32_t>(target_tile_ntt_stride, 2);
          constexpr uint32_t tile_N =
              block_dim.x / (tile_ntt_stride / load_words);

          dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N, batch_size);
          return cudastf::cuda_kernel_desc{
              copy_for_ntt<l_word, r_word, load_words, tile_N, tile_ntt_stride,
                           merged>,
              grid_dim,
              block_dim,
              0,
              ntt_res_ptr,
              lwe_data_ptr,
              rlwe_scale,
              modulus,
              N,
              fldim,
              ntt_stride,
              current_ntt_column_size};
        } else if (ntt_stride % 8 != 0) {
          constexpr uint32_t load_words =
              std::min<uint32_t>(target_load_words, 4);
          constexpr uint32_t tile_ntt_stride =
              std::min<uint32_t>(target_tile_ntt_stride, 4);
          constexpr uint32_t tile_N =
              block_dim.x / (tile_ntt_stride / load_words);

          dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N, batch_size);
          return cudastf::cuda_kernel_desc{
              copy_for_ntt<l_word, r_word, load_words, tile_N, tile_ntt_stride,
                           merged>,
              grid_dim,
              block_dim,
              0,
              ntt_res_ptr,
              lwe_data_ptr,
              rlwe_scale,
              modulus,
              N,
              fldim,
              ntt_stride,
              current_ntt_column_size};
        } else if (ntt_stride % 16 != 0) {
          constexpr uint32_t load_words =
              std::min<uint32_t>(target_load_words, 8);
          constexpr uint32_t tile_ntt_stride =
              std::min<uint32_t>(target_tile_ntt_stride, 8);
          constexpr uint32_t tile_N =
              block_dim.x / (tile_ntt_stride / load_words);

          dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N, batch_size);
          return cudastf::cuda_kernel_desc{
              copy_for_ntt<l_word, r_word, load_words, tile_N, tile_ntt_stride,
                           merged>,
              grid_dim,
              block_dim,
              0,
              ntt_res_ptr,
              lwe_data_ptr,
              rlwe_scale,
              modulus,
              N,
              fldim,
              ntt_stride,
              current_ntt_column_size};
        } else { // ntt_stride is a multiple of 16
          constexpr uint32_t load_words =
              std::min<uint32_t>(target_load_words, 16);
          constexpr uint32_t tile_ntt_stride =
              std::min<uint32_t>(target_tile_ntt_stride, 16);
          constexpr uint32_t tile_N =
              block_dim.x / (tile_ntt_stride / load_words);

          dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N, batch_size);
          return cudastf::cuda_kernel_desc{
              copy_for_ntt<l_word, r_word, load_words, tile_N, tile_ntt_stride,
                           merged>,
              grid_dim,
              block_dim,
              0,
              ntt_res_ptr,
              lwe_data_ptr,
              rlwe_scale,
              modulus,
              N,
              fldim,
              ntt_stride,
              current_ntt_column_size};
        }
      };
  // Now, perform NTT
  ntt_inplace(ntt_res, batch_size * ntt_stride_);
}

template <typename l_word, typename r_word>
template <bool accumulate, int second_dim_overwrite_ct>
void RLWECompress<l_word, r_word>::gemm_worker(
    logical_data_t<cudastf::slice<r_word>> &res,
    logical_data_t<cudastf::slice<r_word>> &a_ntt,
    logical_data_t<cudastf::slice<r_word>> &conversion_data, size_t batch_size,
    size_t batch_idx) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  cudastf::cuda_safe_call(cudaSetDevice(device_id_));

  constexpr uint32_t copy_granularity_bytes = 32;
  constexpr uint32_t copy_granularity_r_words =
      copy_granularity_bytes / sizeof(r_word);
  constexpr uint32_t copy_group = copy_granularity_bytes / 16;

  if (ntt_stride_ % 64 != 0 && !is_pow2(ntt_stride_)) {
    std::cout << "Warning: ntt_stride_ % 64 != 0 in gemm_worker, may show "
                 "suboptimal performance."
              << std::endl;
  }
  if (lwe_->n_ % 8 != 0) {
    throw std::invalid_argument(
        "RLWECompress::gemm_worker: lwe_->n_ must be divisible by 8");
  }

  lwe_ann::barrett_util<r_word> barrett(rlwe_->modulus_);
  auto kernel_desc_function =
      [n = lwe_->n_, N = rlwe_->N_, batch_size = batch_size,
       ntt_stride = this->ntt_stride_, modulus = rlwe_->modulus_,
       barrett_k = barrett.barrett_k_double_,
       barrett_rcp = barrett.barrett_reciprocal_double_, batch_idx = batch_idx](
          cudastf::slice<r_word> res, cudastf::slice<const r_word> a_ntt_res,
          cudastf::slice<const r_word> conversion_data) {
        if (res.size() != N * batch_size * ntt_stride) {
          throw std::invalid_argument(
              "RLWECompress::gemm_worker: res.size() must be equal "
              "to N * batch_size * ntt_stride");
        }
        if (a_ntt_res.size() != N * n * ntt_stride) {
          throw std::invalid_argument(
              "RLWECompress::gemm_worker: a_ntt_res.size() must "
              "be equal to N * n * ntt_stride");
        }
        const r_word *conversion_data_ptr = conversion_data.data_handle();
        if constexpr (second_dim_overwrite_ct != -1) {
          conversion_data_ptr += batch_idx * N * n;
        }

        // if (conversion_data.size() != N * batch_size * n) {
        //   throw std::invalid_argument(
        //       "RLWECompress::gemm_worker: conversion_data.size() "
        //       "must be equal to N * batch_size * n");
        // }

        // The following is a somewhat arbitrary solution.
        if (ntt_stride % 2 != 0) {
          constexpr uint32_t tile_ntt_stride = 1;
          constexpr uint32_t tile_batch_size = 64;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 1;
          constexpr uint32_t th_y = 64;
          constexpr uint32_t th_z = copy_group;
          GEMM_WORKER_BODY;
        } else if (ntt_stride % 4 != 0) {
          constexpr uint32_t tile_ntt_stride = 2;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 2;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            constexpr uint32_t th_y = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            constexpr uint32_t th_y = 64;
            GEMM_WORKER_BODY;
          }
        } else if (ntt_stride % 8 != 0) {
          constexpr uint32_t tile_ntt_stride = 4;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 4;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 16) {
            constexpr uint32_t tile_batch_size = 16;
            constexpr uint32_t th_y = 16;
            GEMM_WORKER_BODY;
          } else if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            constexpr uint32_t th_y = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            constexpr uint32_t th_y = 32;
            GEMM_WORKER_BODY;
          }
        } else if (ntt_stride % 16 != 0) {
          constexpr uint32_t tile_ntt_stride = 8;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 8;
          constexpr uint32_t th_y = 16;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 16) {
            constexpr uint32_t tile_batch_size = 16;
            GEMM_WORKER_BODY;
          } else if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            GEMM_WORKER_BODY;
          }
        } else if (ntt_stride % 32 != 0) {
          constexpr uint32_t tile_ntt_stride = 16;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 16;
          constexpr uint32_t th_y = 8;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 16) {
            constexpr uint32_t tile_batch_size = 16;
            GEMM_WORKER_BODY;
          } else if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            GEMM_WORKER_BODY;
          }
        } else if (ntt_stride % 64 != 0) {
          constexpr uint32_t tile_ntt_stride = 32;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 16;
          constexpr uint32_t th_y = 8;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 16) {
            constexpr uint32_t tile_batch_size = 16;
            GEMM_WORKER_BODY;
          } else if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            GEMM_WORKER_BODY;
          }
        } else {
          constexpr uint32_t tile_ntt_stride = 64;
          constexpr uint32_t tile_n = 8;
          constexpr uint32_t th_x = 16;
          constexpr uint32_t th_y = 8;
          constexpr uint32_t th_z = copy_group;
          if (batch_size <= 16) {
            constexpr uint32_t tile_batch_size = 16;
            GEMM_WORKER_BODY;
          } else if (batch_size <= 32) {
            constexpr uint32_t tile_batch_size = 32;
            GEMM_WORKER_BODY;
          } else {
            constexpr uint32_t tile_batch_size = 64;
            GEMM_WORKER_BODY;
          }
        }
      };
  if constexpr (accumulate) {
    cudastf_ctx
            ->cuda_kernel(
                res.rw(cudastf::data_place::device(device_id_)),
                a_ntt.read(cudastf::data_place::device(device_id_)),
                conversion_data.read(cudastf::data_place::device(device_id_)))
            .set_symbol("conversion_gemm_b")
            ->*kernel_desc_function;
  } else {
    cudastf_ctx
            ->cuda_kernel(
                res.write(cudastf::data_place::device(device_id_)),
                a_ntt.read(cudastf::data_place::device(device_id_)),
                conversion_data.read(cudastf::data_place::device(device_id_)))
            .set_symbol("conversion_gemm_a")
            ->*kernel_desc_function;
  }
}

template <typename l_word, typename r_word>
template <typename plain_word>
void RLWECompress<l_word, r_word>::rlwe_reinterpret_worker(
    std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
    logical_data_t<cudastf::slice<r_word>> &src, size_t batch_size,
    size_t src_idx, bool is_b_data) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  // size_t num_plain_word_chunks =
  //     div_ceil<size_t>(log2_ceil(rlwe_->modulus_) - rlwe_round_shift_bits_,
  //                      sizeof(plain_word) * 8);
  size_t num_plain_word_chunks = get_num_plain_word_chunks<plain_word>();
  size_t padded_num_ntt_columns = pad_by<size_t>(ntt_column_size_, rlwe_->N_);
  size_t res_size_per_batch =
      2 * padded_num_ntt_columns * ntt_stride_ * num_plain_word_chunks;

  bool first = (res.size() == 0);

  for (size_t i = 0; i < batch_size; i++) {
    if (first) {
      res.emplace_back(
          cudastf_ctx
              ->logical_data(cudastf::shape_of<cudastf::slice<plain_word>>(
                  res_size_per_batch))
              .set_symbol("reinterpreted_res_" + std::to_string(i) + "_GPU_" +
                          std::to_string(device_id_)));
    }
    cudastf_ctx->cuda_kernel(
        res.at(i).write(cudastf::data_place::device(device_id_)),
        src.read(cudastf::data_place::device(device_id_)))
            ->*
        [N = rlwe_->N_, batch_size = batch_size, ntt_stride = this->ntt_stride_,
         modulus = rlwe_->modulus_, round_shift_bits = rlwe_round_shift_bits_,
         num_plain_word_chunks = num_plain_word_chunks,
         res_size_per_batch = res_size_per_batch, src_idx = src_idx,
         batch_idx = i, is_b_data = is_b_data](auto res, auto src) {
          // Check input sizes
          if (res.size() != res_size_per_batch) {
            throw std::invalid_argument(
                "RLWECompress::rlwe_reinterpret_worker: res.size() must be "
                "equal to res_size_per_batch");
          }
          if (src.size() != N * batch_size * ntt_stride) {
            throw std::invalid_argument(
                "RLWECompress::rlwe_reinterpret_worker: src.size() must be "
                "equal to N * batch_size * ntt_stride");
          }

          auto res_ptr = res.data_handle() +
                         src_idx * N * ntt_stride * num_plain_word_chunks;
          if (is_b_data) {
            res_ptr += (res_size_per_batch / 2);
          }
          auto src_ptr = src.data_handle() + batch_idx * N * ntt_stride;

          // Default
          constexpr dim3 block_dim(256);

          // We want to load 16 bytes at a time.
          constexpr uint32_t target_load_words = 16 / sizeof(plain_word);
          // At a warp level, we want to load 32 bytes at a time.
          constexpr uint32_t target_tile_ntt_stride = 32 / sizeof(plain_word);

          if (ntt_stride % 2 != 0) {
            constexpr uint32_t load_words = 1;
            constexpr uint32_t tile_ntt_stride = 1;
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          } else if (ntt_stride % 4 != 0) {
            constexpr uint32_t load_words =
                std::min<uint32_t>(target_load_words, 2);
            constexpr uint32_t tile_ntt_stride =
                std::min<uint32_t>(target_tile_ntt_stride, 2);
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          } else if (ntt_stride % 8 != 0) {
            constexpr uint32_t load_words =
                std::min<uint32_t>(target_load_words, 4);
            constexpr uint32_t tile_ntt_stride =
                std::min<uint32_t>(target_tile_ntt_stride, 4);
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          } else if (ntt_stride % 16 != 0) {
            constexpr uint32_t load_words =
                std::min<uint32_t>(target_load_words, 8);
            constexpr uint32_t tile_ntt_stride =
                std::min<uint32_t>(target_tile_ntt_stride, 8);
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          } else if (ntt_stride % 32 != 0) {
            constexpr uint32_t load_words =
                std::min<uint32_t>(target_load_words, 16);
            constexpr uint32_t tile_ntt_stride =
                std::min<uint32_t>(target_tile_ntt_stride, 16);
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          } else {
            constexpr uint32_t load_words =
                std::min<uint32_t>(target_load_words, 32);
            constexpr uint32_t tile_ntt_stride =
                std::min<uint32_t>(target_tile_ntt_stride, 32);
            constexpr uint32_t tile_N =
                block_dim.x / (tile_ntt_stride / load_words);

            dim3 grid_dim(ntt_stride / tile_ntt_stride, N / tile_N);
            return cudastf::cuda_kernel_desc{
                rlwe_reinterpret<r_word, plain_word, load_words, tile_N,
                                 tile_ntt_stride>,
                grid_dim,
                block_dim,
                0,
                res_ptr,
                src_ptr,
                N,
                ntt_stride,
                modulus,
                round_shift_bits,
                num_plain_word_chunks};
          }
        };
  }
}

template <typename l_word, typename r_word>
void RLWECompress<l_word, r_word>::multiply_conversion_ciphertexts(
    std::vector<logical_data_t<cudastf::slice<r_word>>> &res_a,
    std::vector<logical_data_t<cudastf::slice<r_word>>> &res_b,
    logical_data_t<cudastf::slice<r_word>> &conversion_a,
    logical_data_t<cudastf::slice<r_word>> &conversion_b,
    logical_data_t<cudastf::slice<l_word>> &lwe_b, size_t lwe_batch_size) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  res_a.clear();
  res_b.clear();

  // We launch a-part kernels first, which may be prepared more early.
  for (size_t i = 0; i < a_ntt_results_.size(); i++) {
    res_a.emplace_back(
        cudastf_ctx
            ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                rlwe_->N_ * ntt_stride_ * lwe_batch_size))
            .set_symbol("compressed_a_" + std::to_string(i)));
    auto &res = res_a.back();
    gemm_worker<false>(res, a_ntt_results_.at(i), conversion_a, lwe_batch_size);
  }

  // Mostly the same as prepare_a_ntt
  size_t ntt_column_index = 0;
  for (size_t i = 0; i < a_ntt_results_.size(); i++) {
    size_t current_ntt_column_size =
        std::min(ntt_column_size_ - ntt_column_index, rlwe_->N_);
    res_b.emplace_back(
        cudastf_ctx
            ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                rlwe_->N_ * ntt_stride_ * lwe_batch_size))
            .set_symbol("compressed_b_" + std::to_string(i)));
    auto &res = res_b.back();
    copy_and_ntt<false>(res, lwe_b, ntt_column_index, current_ntt_column_size,
                        lwe_batch_size, true);
    gemm_worker<true>(res, a_ntt_results_.at(i), conversion_b, lwe_batch_size);
    ntt_column_index += rlwe_->N_;
  }
}

template <typename l_word, typename r_word>
template <typename plain_word>
void RLWECompress<l_word, r_word>::
    multiply_conversion_ciphertexts_with_reinterpret(
        std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
        logical_data_t<cudastf::slice<r_word>> &conversion_a,
        logical_data_t<cudastf::slice<r_word>> &conversion_b,
        logical_data_t<cudastf::slice<l_word>> &lwe_b, size_t lwe_batch_size) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  res.clear();

  // We launch a-part kernels first, which may be prepared more early.
  for (size_t i = 0; i < a_ntt_results_.size(); i++) {
    auto tmp = cudastf_ctx
                   ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                       rlwe_->N_ * ntt_stride_ * lwe_batch_size))
                   .set_symbol("compressed_a_" + std::to_string(i) + "_GPU_" +
                               std::to_string(device_id_));
    gemm_worker<false>(tmp, a_ntt_results_.at(i), conversion_a, lwe_batch_size);
    inverse_ntt_inplace(tmp, ntt_stride_ * lwe_batch_size);
    rlwe_reinterpret_worker<plain_word>(res, tmp, lwe_batch_size, i, false);
  }

  // Mostly the same as prepare_a_ntt + gemm + reinterpret
  size_t ntt_column_index = 0;
  for (size_t i = 0; i < a_ntt_results_.size(); i++) {
    size_t current_ntt_column_size =
        std::min(ntt_column_size_ - ntt_column_index, rlwe_->N_);
    auto tmp = cudastf_ctx
                   ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                       rlwe_->N_ * ntt_stride_ * lwe_batch_size))
                   .set_symbol("compressed_b_" + std::to_string(i) + "_GPU_" +
                               std::to_string(device_id_));
    copy_and_ntt<false>(tmp, lwe_b, ntt_column_index, current_ntt_column_size,
                        lwe_batch_size, true);
    gemm_worker<true>(tmp, a_ntt_results_.at(i), conversion_b, lwe_batch_size);
    inverse_ntt_inplace(tmp, ntt_stride_ * lwe_batch_size);
    rlwe_reinterpret_worker<plain_word>(res, tmp, lwe_batch_size, i, true);
    ntt_column_index += rlwe_->N_;
  }
}

template <typename l_word, typename r_word>
template <typename plain_word>
void RLWECompress<l_word, r_word>::do_everything_per_batch(
    std::vector<logical_data_t<cudastf::slice<plain_word>>> &res,
    logical_data_t<cudastf::slice<l_word>> &merged_data,
    logical_data_t<cudastf::slice<r_word>> &conversion_a,
    logical_data_t<cudastf::slice<r_word>> &conversion_b, size_t batch_idx) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  // Perform NTT on the merged data
  std::vector<logical_data_t<cudastf::slice<r_word>>> ntt_results;
  for (size_t i = 0, ntt_column_index = 0; ntt_column_index < ntt_column_size_;
       i++, ntt_column_index += rlwe_->N_) {
    size_t current_ntt_column_size =
        std::min(ntt_column_size_ - ntt_column_index, rlwe_->N_);
    ntt_results.emplace_back(
        cudastf_ctx
            ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                rlwe_->N_ * lwe_->n_ * ntt_stride_))
            .set_symbol("a_ntt_buffer_" + std::to_string(i)));
    copy_and_ntt<true>(ntt_results.at(i), merged_data, ntt_column_index,
                       current_ntt_column_size, lwe_->n_, false);
  }

  res.clear();
  // We launch a-part kernels first, which may be prepared more early.
  for (size_t i = 0; i < ntt_results.size(); i++) {
    auto tmp = cudastf_ctx
                   ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                       rlwe_->N_ * ntt_stride_))
                   .set_symbol("compressed_a_" + std::to_string(i) + "_GPU_" +
                               std::to_string(device_id_));
    gemm_worker<false, 0>(tmp, ntt_results.at(i), conversion_a, 1, batch_idx);
    inverse_ntt_inplace(tmp, ntt_stride_);
    rlwe_reinterpret_worker<plain_word>(res, tmp, 1, i, false);
  }

  // Mostly the same as prepare_a_ntt + gemm + reinterpret
  for (size_t i = 0; i < ntt_results.size(); i++) {
    auto tmp = cudastf_ctx
                   ->logical_data(cudastf::shape_of<cudastf::slice<r_word>>(
                       rlwe_->N_ * ntt_stride_))
                   .set_symbol("compressed_b_" + std::to_string(i) + "_GPU_" +
                               std::to_string(device_id_));
    gemm_worker<false, 1>(tmp, ntt_results.at(i), conversion_b, 1, batch_idx);
    inverse_ntt_inplace(tmp, ntt_stride_);
    rlwe_reinterpret_worker<plain_word>(res, tmp, 1, i, true);
  }
}

template <typename l_word, typename r_word>
void RLWECompress<l_word, r_word>::ntt_inplace(
    logical_data_t<cudastf::slice<r_word>> &ntt_inout, size_t num_polys) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  cudastf::cuda_safe_call(cudaSetDevice(device_id_));
  if (num_polys > max_ntt_num_polys_) {
    throw std::invalid_argument(
        "RLWECompress::ntt_inplace: num_polys must be less than " +
        std::to_string(max_ntt_num_polys_));
  }
  // Get resources before lambda
  int32_t devid = device_id_;
  uint64_t* twiddles = ntt_twiddles::get_twiddles(rlwe_->N_, devid, -1);
  uint64_t* psi_powers = ntt_twiddles::get_psi_powers(rlwe_->N_, devid);
  constexpr uint64_t modulus = 2251799780524033ULL;
  constexpr uint64_t mu = 18014398774771707ULL;
  
  cudastf_ctx
          ->cuda_kernel_chain(
              ntt_inout.rw(cudastf::data_place::device(device_id_)))
          .set_symbol("ntt_inplace")
          ->*
      [num_polys = num_polys, N = rlwe_->N_, twiddles, psi_powers, modulus, mu](auto ntt_inout) {
        if (num_polys * N != ntt_inout.size()) {
          throw std::invalid_argument(
              "RLWECompress::ntt_inplace: num_polys * N must be equal to "
              "ntt_inout.size()");
        }

        // Forward: negacyclic_scale → NTT
        if (N != 2048) {
          throw std::invalid_argument("Only N=2048 is supported, got: " + std::to_string(N));
        }
        
        return std::vector<cudastf::cuda_kernel_desc>{
            {negacyclic_scale_kernel_with_powers, num_polys, 1024, size_t(0),
             ntt_inout.data_handle(), psi_powers, modulus, mu, size_t(2048), size_t(num_polys)},
            {ker_code0_2048, num_polys, 1024, size_t(2 * 2048 * sizeof(uint64_t)),
             ntt_inout.data_handle(), ntt_inout.data_handle(), modulus, twiddles, mu}
        };
      };
}

template <typename l_word, typename r_word>
void RLWECompress<l_word, r_word>::inverse_ntt_inplace(
    logical_data_t<cudastf::slice<r_word>> &ntt_inout, size_t num_polys) {
  auto cudastf_ctx = lwe_->cudastf_ctx_;

  cudastf::cuda_safe_call(cudaSetDevice(device_id_));
  if (num_polys > max_ntt_num_polys_) {
    throw std::invalid_argument(
        "RLWECompress::inverse_ntt_inplace: num_polys must be less than " +
        std::to_string(max_ntt_num_polys_));
  }
  // Get resources BEFORE lambda
  int32_t devid = device_id_;
  uint64_t* twiddles = ntt_twiddles::get_twiddles(rlwe_->N_, devid, 1);
  uint64_t* psi_inv_powers = ntt_twiddles::get_psi_inv_powers(rlwe_->N_, devid);
  uint64_t inv_n_val = ntt_twiddles::get_inv_N(rlwe_->N_);
  constexpr uint64_t modulus = 2251799780524033ULL;
  constexpr uint64_t mu = 18014398774771707ULL;
  
  cudastf_ctx
          ->cuda_kernel_chain(
              ntt_inout.rw(cudastf::data_place::device(device_id_)))
          .set_symbol("inverse_ntt_inplace")
          ->*
      [num_polys = num_polys, N = rlwe_->N_, twiddles, psi_inv_powers, inv_n_val, modulus, mu](auto ntt_inout) {
        if (num_polys * N != ntt_inout.size()) {
          throw std::invalid_argument(
              "RLWECompress::inverse_ntt_inplace: num_polys * N must be equal to "
              "ntt_inout.size()");
        }

        // bit-reverse → inv_NTT → bit-reverse → normalize → descale
        if (N != 2048) {
          throw std::invalid_argument("Only N=2048 is supported for inverse NTT, got: " + std::to_string(N));
        }
        
        size_t shared_mem = 2 * N * sizeof(uint64_t);
        size_t block_threads = 1024;
        uint64_t* data_ptr = ntt_inout.data_handle();
        
        return std::vector<cudastf::cuda_kernel_desc>{
            {bit_reverse_kernel, num_polys, block_threads, size_t(N * sizeof(uint64_t)),
             data_ptr, size_t(N), size_t(num_polys)},
            {ker_code0_2048, num_polys, 1024, shared_mem,
             data_ptr, data_ptr, modulus, twiddles, mu},
            {bit_reverse_kernel, num_polys, block_threads, size_t(N * sizeof(uint64_t)),
             data_ptr, size_t(N), size_t(num_polys)},
            {scale_inv_ntt_kernel, num_polys, block_threads, size_t(0),
             data_ptr, inv_n_val, modulus, mu, size_t(N), size_t(num_polys)},
            {negacyclic_descale_kernel_with_powers, num_polys, block_threads, size_t(0),
             data_ptr, psi_inv_powers, modulus, mu, size_t(N), size_t(num_polys)}
        };
      };
}

template class RLWECompress<int32_t, uint64_t>;
template void RLWECompress<int32_t, uint64_t>::
    multiply_conversion_ciphertexts_with_reinterpret<int8_t>(
        std::vector<logical_data_t<cudastf::slice<int8_t>>> &,
        logical_data_t<cudastf::slice<uint64_t>> &,
        logical_data_t<cudastf::slice<uint64_t>> &,
        logical_data_t<cudastf::slice<int32_t>> &, size_t);
template void RLWECompress<int32_t, uint64_t>::do_everything_per_batch<int8_t>(
    std::vector<logical_data_t<cudastf::slice<int8_t>>> &,
    logical_data_t<cudastf::slice<int32_t>> &,
    logical_data_t<cudastf::slice<uint64_t>> &,
    logical_data_t<cudastf::slice<uint64_t>> &, size_t);
} // namespace lwe_ann::detail

