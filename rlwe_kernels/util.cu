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

#include "util.h"

template <typename T, int GRID_SIZE, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void memset_kernel(T *dst, size_t N)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    while (i < N) {
       dst[i] = 0;
       i += BLOCK_SIZE * GRID_SIZE;
    }
}

template <typename T>
void memset_append(std::vector<cuda_kernel_desc> &descs, T *dst, size_t N)
{
    descs.emplace_back(memset_kernel<T, 2048, 256>, 2048, 256, 0, dst, N);
}

template <typename T, int GRID_SIZE, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void memcpy_kernel(T *dst, const T *src, size_t cnt)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    while (i < cnt) {
        dst[i] = src[i];
        i += BLOCK_SIZE * GRID_SIZE;
    }
}

template <typename T>
void memcpy_append(std::vector<cuda_kernel_desc> &descs, T *dst, const T *src, size_t cnt)
{
    descs.emplace_back(memcpy_kernel<T, 2048, 256>, 2048, 256, 0, dst, src, cnt);
}

template <typename T>
void do_memcpy_task(stf_context_t &ctx, logical_data_t<slice<T>>& dst, const logical_data_t<slice<T>>& src)
{
    ctx.cuda_kernel_chain(dst.write(), src.read()).set_symbol("memcpy")->*[=](auto dst, auto src) {
        std::vector<cuda_kernel_desc> result;
        result.emplace_back(memcpy_kernel<T, 2048, 256>, 2048, 256, 0, dst.data_handle(), src.data_handle(), dst.size());
        return result;
    };
}

/* Explicit instantiations with uint64_t */
template void do_memcpy_task<uint64_t>(stf_context_t &ctx, logical_data_t<slice<uint64_t>>& dst, const logical_data_t<slice<uint64_t>>& src);

template __global__ void memset_kernel<uint64_t, 2048, 256>(uint64_t *dst, size_t N);

template void memset_append<uint64_t>(std::vector<cuda_kernel_desc> &descs, uint64_t *dst, size_t N);
template void memcpy_append(std::vector<cuda_kernel_desc> &descs, uint64_t *dst, const uint64_t *src, size_t cnt);

template <typename T>
void launch_memcpy(cudaStream_t stream, T *dst, const T *src, size_t cnt)
{
    memcpy_kernel<T, 2048, 256><<<2048, 256, 0, stream>>>(dst, src, cnt);
}

template void launch_memcpy(cudaStream_t stream, uint64_t *dst, const uint64_t *src, size_t cnt);
