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

#include <cuda/experimental/stf.cuh>

namespace cudastf=cuda::experimental::stf;

namespace cuda::experimental::stf {

/* If max_entries is 0, all values are displayed, otherwise only the first entries are shown */
template <typename Ctx_t, typename T>
void compare_ld(Ctx_t &ctx,
                cudastf::logical_data<cudastf::slice<T>> ld1,
                cudastf::logical_data<cudastf::slice<T>> ld2,
                size_t max_entries = 0,
                size_t offset = 0)
{
      ctx.task(cudastf::exec_place::host(), ld1.read(), ld2.read())->*[=](cudaStream_t stream, auto buf1, auto buf2) {
           cudaStreamSynchronize(stream);

           // Cap the number of elements if necessary
           size_t cnt1 = buf1.size();
           size_t cnt2 = buf2.size();

           bool passed = (cnt1 == cnt2);

           if(passed)
           {
               if (max_entries > 0 && max_entries < cnt1) cnt1 = max_entries;

               for (size_t i = offset; i < offset + cnt1; i++)
               {
                    passed &= (buf1[i] == buf2[i]);
               }
           }

           printf("%s\n", passed ? "passed" : "failed");
      };
};

} // end namespace cuda::experimental::stf
