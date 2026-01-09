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

#include <complex>
#include <memory>
#include <vector>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include "ntt_twiddles.h"
#include "ntt/ntt_helpers.cu"
#include "ntt/ntt2048x51.cu"
#include "rlwe_cudastf.h"

namespace cudastf=cuda::experimental::stf;
using namespace cudastf;

#define CUDA_CHECK(ans) { gpu_checkAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpu_checkAssert(cudaError_t code, const char* arg, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA_CHECK: %s (%d), %s at %s:%d\n", cudaGetErrorString(code), code, arg, file, line);
        if (abort) exit(code);
    }
}

template<typename ValueType>
::std::vector<cudastf::cuda_kernel_desc> ntt_backend(void* idata, void* odata, int direction, unsigned int nbatches, unsigned int transform_size,
                    const uint8_t* moduli)
{
    int32_t devid = cuda_try<cudaGetDevice>();
    
    constexpr uint64_t modulus = 2251799780524033ULL;
    constexpr uint64_t mu = 18014398774771707ULL;
    
    uint64_t* idata_u64 = reinterpret_cast<uint64_t*>(idata);
    uint64_t* odata_u64 = reinterpret_cast<uint64_t*>(odata);
    uint64_t* twiddles = ntt_twiddles::get_twiddles(transform_size, devid, direction);
    
    if (direction == -1) {
        // Forward: negacyclic_scale → NTT
        uint64_t* psi_powers = ntt_twiddles::get_psi_powers(transform_size, devid);
        
        if (transform_size != 2048) {
            fprintf(stderr, "Error: Only N=2048 is supported, got %u\n", transform_size);
            abort();
        }
        
        return ::std::vector<cudastf::cuda_kernel_desc>{
            {negacyclic_scale_kernel_with_powers, nbatches, 1024, size_t(0),
             idata_u64, psi_powers, modulus, mu, size_t(2048), size_t(nbatches)},
            {ker_code0_2048, nbatches, 1024, size_t(2 * 2048 * sizeof(uint64_t)),
             idata_u64, odata_u64, modulus, twiddles, mu}
        };
    } else {
        // bit-reverse → inv_NTT → bit-reverse → normalize → descale
        uint64_t* psi_inv_powers = ntt_twiddles::get_psi_inv_powers(transform_size, devid);
        uint64_t inv_n = ntt_twiddles::get_inv_N(transform_size);
        
        if (transform_size != 2048) {
            fprintf(stderr, "Error: Only N=2048 is supported, got %u\n", transform_size);
            abort();
        }
        
        size_t block_threads = 1024;
        
        return ::std::vector<cudastf::cuda_kernel_desc>{
            // Step 1: Bit-reverse input (bit-reversed → normal order)
            {bit_reverse_kernel, nbatches, block_threads, size_t(2048 * sizeof(uint64_t)),
             idata_u64, size_t(2048), size_t(nbatches)},
            // Step 2: Inverse NTT kernel with inverse twiddles
            {ker_code0_2048, nbatches, 1024, size_t(2 * 2048 * sizeof(uint64_t)),
             idata_u64, odata_u64, modulus, twiddles, mu},
            // Step 3: Bit-reverse output (bit-reversed → normal order)
            {bit_reverse_kernel, nbatches, block_threads, size_t(2048 * sizeof(uint64_t)),
             odata_u64, size_t(2048), size_t(nbatches)},
            // Step 4: Normalize by N^{-1}
            {scale_inv_ntt_kernel, nbatches, block_threads, size_t(0),
             odata_u64, inv_n, modulus, mu, size_t(2048), size_t(nbatches)},
            // Step 5: Negacyclic descaling by ψ^{-i}
            {negacyclic_descale_kernel_with_powers, nbatches, block_threads, size_t(0),
             odata_u64, psi_inv_powers, modulus, mu, size_t(2048), size_t(nbatches)}
        };
    }
}
