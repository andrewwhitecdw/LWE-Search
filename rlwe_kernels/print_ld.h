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

#include "rlwe_cudastf.h"

namespace cudastf=cuda::experimental::stf;

namespace cuda::experimental::stf {

/* If max_entries is 0, all values are displayed, otherwise only the first entries are shown */
template <typename Ctx_t, typename T>
void print_ld(Ctx_t &ctx, logical_data_t<cudastf::slice<T>> ld, ::std::string name, size_t max_entries = 0, size_t offset = 0) {
      size_t num_elements =  ld.shape().size();

      printf("**********************************************\n");
      printf("Dumping data : %s (%ld elems) starting at offset %ld\n", name.c_str(), num_elements, offset);
      ctx.task(cudastf::exec_place::host(), ld.read())->*[max_entries, offset](cudaStream_t stream, auto buf) {
           cudaStreamSynchronize(stream);

           // Cap the number of elements if necessary
           size_t cnt = buf.size();
           if (max_entries > 0 && max_entries < cnt) cnt = max_entries;

           for (size_t i = offset; i < offset + cnt; i++)
           {
                 printf("[%ld] 0x%lx\n", i, buf(i));
           }
           printf("\n");
      };
      printf("**********************************************\n");
};

} // end namespace cuda::experimental::stf
