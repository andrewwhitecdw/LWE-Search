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

#include "rlwe_cudastf.h"
#include "rlwe_arith.h"

namespace cudastf=cuda::experimental::stf;
using namespace cudastf;

template <typename T, int, int>
__global__ void memset_kernel(T *dst, size_t N);

template <typename T, int, int>
__global__ void memcpy_kernel(T *dst, const T *src, size_t cnt);

template <typename T>
void do_memcpy_task(stf_context_t &ctx, logical_data_t<slice<T>>& dst, const logical_data_t<slice<T>>& src);

template <typename T>
void memset_append(std::vector<cuda_kernel_desc> &descs, T *dst, size_t N);

template <typename T>
void memcpy_append(std::vector<cuda_kernel_desc> &descs, T *dst, const T *src, size_t cnt);

template <typename T>
void launch_memcpy(cudaStream_t stream, T *dst, const T *src, size_t cnt);

void ld_pack(stf_context_t &ctx, logical_data_t<slice<uint64_t>> &dst, ::std::vector<logical_data_t<slice<uint64_t>>> &src);
void ld_unpack(stf_context_t &ctx, ::std::vector<logical_data_t<slice<uint64_t>>> &dst, logical_data_t<slice<uint64_t>> &src);
