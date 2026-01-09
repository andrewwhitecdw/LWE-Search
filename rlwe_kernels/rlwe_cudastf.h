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

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"
#include "cuda/experimental/__stf/utility/run_once.cuh"
#include "cuda/experimental/stf.cuh"

#include <algorithm>

namespace cudastf=cuda::experimental::stf;


#ifdef USE_STACKABLE
template <typename T>
using logical_data_t = cudastf::stackable_logical_data<T>;

using stf_context_t = cudastf::stackable_ctx;

inline bool disable_stackable() {
    static const bool disabled = [] {
        const char* env_val = std::getenv("DISABLE_STACKABLE");
        return env_val && env_val[0] == '1';
    }();
    return disabled;
}

#define CTX_PUSH(ctx)              \
    do {                           \
        if (!disable_stackable())  \
            (ctx).push();          \
    } while (0)

#define CTX_POP(ctx)               \
    do {                           \
        if (!disable_stackable())  \
            (ctx).pop();           \
    } while (0)

#else
template <typename T>
using logical_data_t = cudastf::logical_data<T>;

using stf_context_t = cudastf::context;

#define CTX_PUSH(ctx) (void)0
#define CTX_POP(ctx) (void)0

#endif


#ifdef USE_STACKABLE
    #define CTX_LOGICAL_DATA_NO_EXPORT(ctx, ...) (ctx).logical_data_no_export(__VA_ARGS__)
#else
    #define CTX_LOGICAL_DATA_NO_EXPORT(ctx, ...) (ctx).logical_data(__VA_ARGS__)
#endif

#define USE_RUN_ONCE

// Helper that creates a constant value and initializes it with a device lambda.
// The additional arguments (ts...) are forwarded to run_once if enabled.
template <typename Func, typename Context, typename Shape, typename... Ts>
auto make_constant(Context &ctx, Shape shape, ::std::string symbol, Func init, Ts&&... ts) {
#ifdef USE_RUN_ONCE
    // Using run_once ensures the initialization code is executed only once.
    auto data = cudastf::run_once(std::forward<Ts>(ts)...)->*[&](auto&&... args) {
        // Create the logical data and assign its symbol.
        auto res = ctx.logical_data(shape).set_symbol(symbol.c_str());
        // Initialize the data via a parallel_for call.
        ctx.parallel_for(res.shape(), res.write())->*::std::forward<Func>(init);
#ifdef USE_STACKABLE
        // After a write access, mark the data as read-only.
        res.set_read_only();
#endif
        return res;
    };
#else
    // Without run_once, directly create the data and run the initialization.
#ifdef USE_STACKABLE
    auto data = ctx.logical_data_no_export(shape).set_symbol(symbol.c_str());
#else
    auto data = ctx.logical_data(shape).set_symbol(symbol.c_str());
#endif
    ctx.parallel_for(data.shape(), data.write())->*::std::forward<Func>(init);
#ifdef USE_STACKABLE
    data.set_read_only();
#endif
#endif
    return data;
}

template <typename T, typename context_t>
inline logical_data_t<cudastf::slice<T>> create_ld_by_copy(context_t &ctx, const T *ptr, size_t cnt) {

    auto res = ctx.logical_data(cudastf::shape_of<cudastf::slice<T>>(cnt));
    std::vector<T> data_copy(ptr, ptr + cnt);
    ctx.host_launch(res.write(cudastf::data_place::managed()))
            .set_symbol("create_ld_by_copy::copy")
            ->*[data = std::move(data_copy)](auto buf) {
                    std::copy(data.begin(), data.end(), buf.data_handle());
                };

    return res;
}


