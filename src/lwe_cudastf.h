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

#include "../rlwe_kernels/rlwe_cudastf.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lwe_ann {

using stf_context_ptr_t = std::shared_ptr<stf_context_t>;

template <typename T>
[[nodiscard]] inline logical_data_t<cudastf::slice<T>>
vector_to_ld(stf_context_ptr_t ctx, const std::vector<T> &data) {
  return create_ld_by_copy(*ctx, data.data(), data.size());
}

template <typename T>
[[nodiscard]] inline logical_data_t<cudastf::slice<T>>
vector_to_ld(stf_context_ptr_t ctx, std::vector<T> &&data) {
  auto res =
      ctx->logical_data(cudastf::shape_of<cudastf::slice<T>>(data.size()));
  ctx->host_launch(res.write(cudastf::data_place::managed()))
          .set_symbol("vector_to_ld")
          ->*[data = std::move(data)](auto buf) {
                std::copy(data.begin(), data.end(), buf.data_handle());
              };
  return res;
}

template <typename T>
inline void __copy_ld_to_host_ptr(stf_context_ptr_t ctx, T *ptr, size_t cnt,
                                  const logical_data_t<cudastf::slice<T>> &ld) {
  ctx->host_launch(ld.read(cudastf::data_place::managed()))
          .set_symbol("__copy_ld_to_host_ptr")
          ->*
      [ptr = ptr, cnt = cnt](auto buf) {
        if (cnt != buf.size()) {
          throw std::invalid_argument(
              "__copy_ld_to_host_ptr failed: size mismatch");
        }
        std::copy(buf.data_handle(), buf.data_handle() + buf.size(), ptr);
      };
  // We have to synchronize here because host_launch is asynchronous
  cudastf::cuda_safe_call(cudaStreamSynchronize(ctx->fence()));
}

} // namespace lwe_ann