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

void ld_pack(stf_context_t &ctx, logical_data_t<slice<uint64_t>> &dst, ::std::vector<logical_data_t<slice<uint64_t>>> &src)
{
    size_t cnt = src.size();
    assert(cnt > 0);

    const size_t part_size = src[0].shape().size();
    assert(dst.shape().size() == cnt * part_size);

    // Create a task that will read all input ciphertexts, and concatenate them into a large buffer
    auto t = ctx.cuda_kernel_chain(dst.write());
    t.set_symbol("pack");
    for (size_t i = 0; i < cnt; i++)
    {
        assert(src[i].shape().size() == part_size);
        t.add_deps(src[i].read());
    }

    t->*[&](auto dst) {
        std::vector<cuda_kernel_desc> descs;
        for (size_t i = 0; i < cnt; i++) {
            auto src = t.get<slice<const uint64_t>>(i+1);
            memcpy_append(descs, &dst(i * part_size), src.data_handle(), part_size);
        }
        return descs;
    };
}

void ld_unpack(stf_context_t &ctx, ::std::vector<logical_data_t<slice<uint64_t>>> &dst, logical_data_t<slice<uint64_t>> &src)
{
    size_t cnt = dst.size();
    assert(cnt > 0);

    const size_t part_size = dst[0].shape().size();
    assert(src.shape().size() == cnt * part_size);

    // Create a task that will read all input ciphertexts, and concatenate them into a large buffer
    auto t = ctx.cuda_kernel_chain(src.read());
    t.set_symbol("unpack");
    for (size_t i = 0; i < cnt; i++)
    {
        t.add_deps(dst[i].write());
    }
    t->*[&](auto src) {
        std::vector<cuda_kernel_desc> descs;
        for (size_t i = 0; i < cnt; i++) {
            auto dst = t.get<slice<uint64_t>>(i+1);
            memcpy_append(descs, dst.data_handle(), &src(i * part_size), part_size);
        }
        return descs;
    };

}
