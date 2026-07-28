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
 
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  }

template <typename T>
void print_matrix(T *matrix, size_t M, size_t N, bool row_major = true) {
  std::cout << "[" << std::endl;
  for (size_t i = 0; i < M; i++) {
    std::cout << "[";
    for (size_t j = 0; j < N; j++) {
      auto value = row_major ? matrix[i * N + j] : matrix[j * M + i];
      std::cout << static_cast<int>(value);
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

template <typename word, typename plain_word, unsigned int tile_m,
          unsigned int tile_n, unsigned int tile_k>
__global__ void naive_gemm_(word *res, const word *lhs, const plain_word *rhs,
                            unsigned int M, unsigned int N, unsigned int K) {
  // blockDim.x == tile_n
  // blockDim.y == tile_m
  static_assert(sizeof(word) == 4, "word must be 4B (currently)");
  static_assert(sizeof(plain_word) == 4 || sizeof(plain_word) == 2 ||
                    sizeof(plain_word) == 1,
                "plain_word must be 4B, 2B, or 1B");

  constexpr unsigned int num_words_per_4B = 4 / sizeof(word);
  constexpr unsigned int num_plain_words_per_4B = 4 / sizeof(plain_word);
  constexpr unsigned int smem_pad = 4; // 4B
  constexpr unsigned int lhs_smem_width = tile_k + smem_pad / sizeof(word);
  constexpr unsigned int lhs_smem_width_4B = lhs_smem_width * sizeof(word) / 4;
  constexpr unsigned int rhs_smem_width =
      tile_k + smem_pad / sizeof(plain_word);
  constexpr unsigned int rhs_smem_width_4B =
      rhs_smem_width * sizeof(plain_word) / 4;

  extern __shared__ char smem[];
  word *smem_lhs = reinterpret_cast<word *>(smem);
  int32_t *smem_lhs_4B = reinterpret_cast<int32_t *>(smem_lhs);
  plain_word *smem_rhs = reinterpret_cast<plain_word *>(
      smem + lhs_smem_width * sizeof(word) * tile_m);
  int32_t *smem_rhs_4B = reinterpret_cast<int32_t *>(smem_rhs);

  lhs += blockIdx.y * tile_m * K;
  rhs += blockIdx.x * tile_n * K;
  res += blockIdx.y * tile_m * N + blockIdx.x * tile_n;

  auto bM = M - blockIdx.y * tile_m;
  auto bN = N - blockIdx.x * tile_n;
  bM = bM > tile_m ? tile_m : bM;
  bN = bN > tile_n ? tile_n : bN;
  constexpr unsigned int n_threads_per_block = tile_n * tile_m;

  word temp_res = 0;

  // We just assume that K is a multiple of tile_k
  // Copying data into shared memory (in 16B chunks)
  constexpr unsigned int lhs_k_load_dim_4B = tile_k / num_words_per_4B;
  constexpr unsigned int rhs_k_load_dim_4B = tile_k / num_plain_words_per_4B;
  unsigned int lhs_total_load_4B = bM * lhs_k_load_dim_4B;
  unsigned int rhs_total_load_4B = bN * rhs_k_load_dim_4B;
  auto tid = threadIdx.x + threadIdx.y * tile_n;

  for (unsigned int k_idx = 0; k_idx < K; k_idx += tile_k) {
    for (unsigned int i = tid; i < lhs_total_load_4B;
         i += n_threads_per_block) {
      auto load_k_idx = i % lhs_k_load_dim_4B;
      auto load_m_idx = i / lhs_k_load_dim_4B;
      smem_lhs_4B[load_m_idx * lhs_smem_width_4B + load_k_idx] =
          *reinterpret_cast<const int32_t *>(
              lhs + (load_m_idx * K + load_k_idx * num_words_per_4B));
    }
    for (unsigned int i = tid; i < rhs_total_load_4B;
         i += n_threads_per_block) {
      auto load_k_idx = i % rhs_k_load_dim_4B;
      auto load_n_idx = i / rhs_k_load_dim_4B;
      smem_rhs_4B[load_n_idx * rhs_smem_width_4B + load_k_idx] =
          *reinterpret_cast<const int32_t *>(
              rhs + (load_n_idx * K + load_k_idx * num_plain_words_per_4B));
    }

    // Move to the next window
    lhs += tile_k;
    rhs += tile_k;

    // Synchronize before using the data in smem_lhs and smem_rhs
    __syncthreads();

    if (threadIdx.y < bM && threadIdx.x < bN) {
      constexpr unsigned int k_iters = tile_k / num_plain_words_per_4B;
      plain_word temp_rhs[num_plain_words_per_4B];
      for (unsigned int i = 0; i < k_iters; i++) {
        *reinterpret_cast<int32_t *>(temp_rhs) =
            smem_rhs_4B[i + threadIdx.x * rhs_smem_width_4B];
        for (unsigned int j = 0; j < num_plain_words_per_4B; j++) {
          temp_res += smem_lhs[threadIdx.y * lhs_smem_width +
                               i * num_plain_words_per_4B + j] *
                      temp_rhs[j];
        }
      }
    }

    // Synchronize before populating the next smem_lhs and smem_rhs
    __syncthreads();
  }

  // store the final result
  if (threadIdx.y < bM && threadIdx.x < bN) {
    res[threadIdx.y * N + threadIdx.x] = temp_res;
  }
}

template <typename word, typename plain_word, unsigned int tile_m,
          unsigned int tile_n, unsigned int tile_k, unsigned int thread_m,
          unsigned int thread_n>
__global__ void gemm_(word *res, const word *lhs, const plain_word *rhs,
                      unsigned int M, unsigned int N, unsigned int K) {
  // blockDim.x == tile_n
  // blockDim.y == tile_m
  static_assert(sizeof(word) == 4, "word must be 4B (currently)");
  static_assert(sizeof(plain_word) == 4 || sizeof(plain_word) == 2 ||
                    sizeof(plain_word) == 1,
                "plain_word must be 4B, 2B, or 1B");

  constexpr unsigned int num_words_per_4B = 4 / sizeof(word);
  constexpr unsigned int num_plain_words_per_4B = 4 / sizeof(plain_word);
  constexpr unsigned int smem_pad = 4; // 4B
  constexpr unsigned int lhs_smem_width = tile_k + smem_pad / sizeof(word);
  constexpr unsigned int lhs_smem_width_4B = lhs_smem_width * sizeof(word) / 4;
  constexpr unsigned int rhs_smem_width =
      tile_k + smem_pad / sizeof(plain_word);
  constexpr unsigned int rhs_smem_width_4B =
      rhs_smem_width * sizeof(plain_word) / 4;

  extern __shared__ char smem[];
  word *smem_lhs = reinterpret_cast<word *>(smem);
  int32_t *smem_lhs_4B = reinterpret_cast<int32_t *>(smem_lhs);
  plain_word *smem_rhs = reinterpret_cast<plain_word *>(
      smem + lhs_smem_width * sizeof(word) * tile_m * thread_m);
  int32_t *smem_rhs_4B = reinterpret_cast<int32_t *>(smem_rhs);

  lhs += blockIdx.y * tile_m * thread_m * K;
  rhs += blockIdx.x * tile_n * thread_n * K;
  res += blockIdx.y * tile_m * thread_m * N + blockIdx.x * tile_n * thread_n;

  auto bM = M - blockIdx.y * tile_m * thread_m;
  auto bN = N - blockIdx.x * tile_n * thread_n;
  bM = bM > tile_m * thread_m ? tile_m * thread_m : bM;
  bN = bN > tile_n * thread_n ? tile_n * thread_n : bN;
  constexpr unsigned int n_threads_per_block = tile_n * tile_m;

  word temp_res[thread_m][thread_n] = {0};

  // We just assume that K is a multiple of tile_k
  // Copying data into shared memory (in 16B chunks)
  constexpr unsigned int lhs_k_load_dim_4B = tile_k / num_words_per_4B;
  constexpr unsigned int rhs_k_load_dim_4B = tile_k / num_plain_words_per_4B;
  unsigned int lhs_total_load_4B = bM * lhs_k_load_dim_4B;
  unsigned int rhs_total_load_4B = bN * rhs_k_load_dim_4B;
  auto tid = threadIdx.x + threadIdx.y * blockDim.x;

  for (unsigned int k_idx = 0; k_idx < K; k_idx += tile_k) {
    for (unsigned int i = tid; i < lhs_total_load_4B;
         i += n_threads_per_block) {
      auto load_k_idx = i % lhs_k_load_dim_4B;
      auto load_m_idx = i / lhs_k_load_dim_4B;
      smem_lhs_4B[load_m_idx * lhs_smem_width_4B + load_k_idx] =
          *reinterpret_cast<const int32_t *>(
              lhs + (load_m_idx * K + load_k_idx * num_words_per_4B));
    }
    for (unsigned int i = tid; i < rhs_total_load_4B;
         i += n_threads_per_block) {
      auto load_k_idx = i % rhs_k_load_dim_4B;
      auto load_n_idx = i / rhs_k_load_dim_4B;
      smem_rhs_4B[load_n_idx * rhs_smem_width_4B + load_k_idx] =
          *reinterpret_cast<const int32_t *>(
              rhs + (load_n_idx * K + load_k_idx * num_plain_words_per_4B));
    }

    // Move to the next window
    lhs += tile_k;
    rhs += tile_k;

    // Synchronize before using the data in smem_lhs and smem_rhs
    __syncthreads();

    constexpr unsigned int k_iters = tile_k / num_plain_words_per_4B;
    plain_word temp_rhs[num_plain_words_per_4B * thread_n];
    word temp_lhs[num_plain_words_per_4B * thread_m];

    for (unsigned int i = 0; i < k_iters; i++) {
      for (unsigned int j = 0; j < thread_n; j++) {
        *(reinterpret_cast<int32_t *>(temp_rhs) + j) =
            smem_rhs_4B[i + (threadIdx.x * thread_m + j) * rhs_smem_width_4B];
      }
      for (unsigned int j = 0; j < thread_m; j++) {
        constexpr unsigned int word_ratio = sizeof(word) / sizeof(plain_word);
        for (unsigned int ii = 0; ii < word_ratio; ii++) {
          *(reinterpret_cast<int32_t *>(temp_lhs) + j * word_ratio + ii) =
              smem_lhs_4B[ii + i * word_ratio +
                          (threadIdx.y * thread_m + j) * lhs_smem_width_4B];
        }
      }

      for (unsigned int j = 0; j < thread_m; j++) {
        for (unsigned int jj = 0; jj < thread_n; jj++) {
          for (unsigned int k = 0; k < num_plain_words_per_4B; k++) {
            temp_res[j][jj] += temp_lhs[j * num_plain_words_per_4B + k] *
                               temp_rhs[jj * num_plain_words_per_4B + k];
          }
        }
      }
    }

    // Synchronize before populating the next smem_lhs and smem_rhs
    __syncthreads();
  }

  // store the final result
  for (unsigned int i = 0; i < thread_m; i++) {
    auto res_m_idx = i + threadIdx.y * thread_m;
    for (unsigned int j = 0; j < thread_n; j++) {
      auto res_n_idx = j + threadIdx.x * thread_n;
      if (res_m_idx < bM && res_n_idx < bN) {
        res[res_m_idx * N + res_n_idx] = temp_res[i][j];
      }
    }
  }
}

int main(int argc, char **argv) {
  unsigned int M = 16;     // batch size
  unsigned int N = 8192;   // 8K clusters
  unsigned int K = 216000; // 2.25K length-96 entries per cluster
  if (argc > 1) {
    M = atoi(argv[1]);
  }
  if (argc > 2) {
    N = atoi(argv[2]);
  }
  if (argc > 3) {
    K = atoi(argv[3]);
  }

  // lhs: 1024 x 2048 matrix
  std::vector<int32_t> lhs(M * K);
  // rhs: 4096 x 2048 matrix
  std::vector<int8_t> rhs(N * K);

  // fill random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dis_lhs(INT32_MIN, INT32_MAX);
  std::uniform_int_distribution<int8_t> dis_rhs(INT8_MIN, INT8_MAX);
  for (auto &x : lhs) {
    x = dis_lhs(gen);
  }
  for (auto &x : rhs) {
    x = dis_rhs(gen);
  }

  // res: 1024 x 4096 matrix
  std::vector<int32_t> res(M * N);

  // Compute res = lhs * rhs^T in CPU
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      for (unsigned int k = 0; k < K; k++) {
        res[i * N + j] += lhs[i * K + k] * rhs[j * K + k];
      }
    }
  }

  int32_t *lhs_gpu = nullptr;
  int8_t *rhs_gpu = nullptr;
  int32_t *res_gpu = nullptr;

  std::cout << "Allocating memory on GPU" << std::endl;
  cudaMalloc(&lhs_gpu, lhs.size() * sizeof(int32_t));
  cudaMalloc(&rhs_gpu, rhs.size() * sizeof(int8_t));
  cudaMalloc(&res_gpu, res.size() * sizeof(int32_t));
  cudaCheckError();

  std::cout << "Copying data to GPU" << std::endl;
  cudaMemcpy(lhs_gpu, lhs.data(), lhs.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_gpu, rhs.data(), rhs.size() * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(res_gpu, res.data(), res.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  constexpr unsigned int tile_m = 8;
  constexpr unsigned int tile_n = 32;
  constexpr unsigned int tile_k = 32;
  constexpr unsigned int thread_m = 2;
  constexpr unsigned int thread_n = 2;

  // Compute res = lhs * rhs^T in GPU
  dim3 grid((N + tile_n * thread_n - 1) / (tile_n * thread_n),
            (M + tile_m * thread_m - 1) / (tile_m * thread_m));
  dim3 block(tile_n, tile_m);
  auto smem = thread_m * tile_m * (tile_k * 4 + 4) +
              thread_n * tile_n * (tile_k * 1 + 4);

  cudaDeviceSynchronize();
  cudaCheckError();
  std::cout << "Launching kernel" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  gemm_<int32_t, int8_t, tile_m, tile_n, tile_k, thread_m, thread_n>
      // naive_gemm_<int32_t, int8_t, tile_m, tile_n, tile_k>
      <<<grid, block, smem>>>(res_gpu, lhs_gpu, rhs_gpu, M, N, K);
  cudaCheckError();
  cudaDeviceSynchronize();
  cudaCheckError();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Kernel execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us" << std::endl;

  std::vector<int32_t> res_gpu_copied(M * N);

  cudaMemcpy(res_gpu_copied.data(), res_gpu, res.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Check if the result is correct
  for (unsigned int i = 0; i < M * N; i++) {
    if (res_gpu_copied[i] != res[i]) {
      std::cerr << "Error at index " << i << ": " << res_gpu_copied[i]
                << " != " << res[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}