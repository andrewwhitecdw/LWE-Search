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

#ifndef NTT_HELPERS_CU_INCLUDED
#define NTT_HELPERS_CU_INCLUDED

#include <cuda_runtime.h>
#include <cstdint>

// Modular multiplication helper (Barrett reduction)
__device__ inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t modulus, uint64_t mu) {
    unsigned __int128 tmp = (unsigned __int128)a * b;
    unsigned __int128 q = ((tmp >> 49) * (unsigned __int128)mu) >> 56;
    tmp -= q * modulus;
    if (tmp >= modulus) tmp -= modulus;
    return (uint64_t)tmp;
}

// Scale all elements by a constant factor (used for N^{-1} scaling in inverse NTT)
__global__ void scale_kernel(uint64_t* data, uint64_t scale_factor, uint64_t modulus, uint64_t mu, size_t N, size_t nbatches) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= nbatches) return;
    
    uint64_t* batch_data = data + batch_idx * N;
    
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        batch_data[i] = mulmod(batch_data[i], scale_factor, modulus, mu);
    }
}

// Negacyclic scaling: multiply element i by psi^i
__global__ void negacyclic_scale_kernel_with_powers(uint64_t* data, uint64_t* psi_powers, uint64_t modulus, uint64_t mu, size_t N, size_t nbatches) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= nbatches) return;
    
    uint64_t* batch_data = data + batch_idx * N;
    
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        batch_data[i] = mulmod(batch_data[i], psi_powers[i], modulus, mu);
    }
}

// Negacyclic descaling: multiply element i by psi^{-i}
__global__ void negacyclic_descale_kernel_with_powers(uint64_t* data, uint64_t* psi_inv_powers, uint64_t modulus, uint64_t mu, size_t N, size_t nbatches) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= nbatches) return;
    
    uint64_t* batch_data = data + batch_idx * N;
    
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        batch_data[i] = mulmod(batch_data[i], psi_inv_powers[i], modulus, mu);
    }
}

// Scale by inverse N (normalization after inverse NTT)
__global__ void scale_inv_ntt_kernel(uint64_t* data, uint64_t inv_N, uint64_t modulus, uint64_t mu, size_t N, size_t nbatches) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= nbatches) return;
    
    uint64_t* batch_data = data + batch_idx * N;
    
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        batch_data[i] = mulmod(batch_data[i], inv_N, modulus, mu);
    }
}

// Bit-reverse kernel: rearranges data from bit-reversed order to normal order (or vice versa)
__global__ void bit_reverse_kernel(uint64_t* data, size_t N, size_t nbatches) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= nbatches) return;
    
    extern __shared__ uint64_t temp[];
    uint64_t* batch_data = data + batch_idx * N;
    
    // Calculate log2(N) for bit-reversal
    size_t log_n = 0;
    size_t temp_n = N;
    while (temp_n > 1) {
        log_n++;
        temp_n >>= 1;
    }
    
    // Load data to shared memory
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        temp[i] = batch_data[i];
    }
    __syncthreads();
    
    // Write back with bit-reversed indices
    for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
        // Compute bit-reversed index
        size_t rev_i = 0;
        for (size_t b = 0; b < log_n; b++) {
            if (i & (1 << b)) {
                rev_i |= (1 << (log_n - 1 - b));
            }
        }
        batch_data[rev_i] = temp[i];
    }
}

#endif // NTT_HELPERS_CU_INCLUDED

