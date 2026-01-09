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

#include <cstddef>
#include <cstdint>

#include "../rlwe_kernels/rlwe_cudastf.h"

namespace lwe_ann {

inline std::size_t get_gemm_m_tile_size(std::size_t m) {
  // This is just a heuristic, may depend on systems
  if (m <= 16) {
    return 16;
  } else if (m <= 128) {
    return 32;
  } else {
    return 64;
  }
}

/**
 * @brief Prepare a GEMM kernel for C = (A * B) % 2^32
 *
 * @param m Number of rows of A / Number of rows of C
 * @param n Number of columns of B / Number of columns of C
 * @param k Number of columns of A / Number of rows of B
 * @param A m x k int32_t matrix (k-major; i.e., transposed)
 * @param B k x n int8_t matrix (k-major; i.e., non-transposed)
 * @param C m x n int32_t matrix (n-major; i.e., transposed)
 * @param round_shift_bits The number of bits to shift the result by.
 *                         If 0, the result is not rounded.
 * @return cudastf::cuda_kernel_desc The kernel descriptor for CUDASTF
 */
cudastf::cuda_kernel_desc get_gemm_kernel_desc(std::size_t m, std::size_t n,
                                               std::size_t k, int32_t const *A,
                                               int8_t const *B, int32_t *C,
                                               size_t round_shift_bits = 0);

} // namespace lwe_ann
