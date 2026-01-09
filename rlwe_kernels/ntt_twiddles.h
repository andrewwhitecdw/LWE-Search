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

#include <stdint.h>
#include <cuda_runtime.h>

namespace ntt_twiddles {

void init_twiddle_factors(uint64_t modulus, size_t poly_degree, int device_id = 0);

// Get twiddle factor pointer for a specific size, device, and direction
// direction: -1 for forward NTT, 1 for inverse NTT
uint64_t* get_twiddles(size_t size, int device_id = 0, int direction = -1);

// Get the inverse modulus factor (N^{-1} mod q) for scaling after inverse NTT
uint64_t get_inv_N(size_t N);

// Get the 2N-th primitive root (psi) for negacyclic scaling
uint64_t get_psi(size_t N);

// Get the inverse of psi for negacyclic descaling
uint64_t get_psi_inv(size_t N);

// Get device pointer to precomputed psi powers for negacyclic scaling
uint64_t* get_psi_powers(size_t N, int device_id = 0);

// Get device pointer to precomputed psi^{-i} powers for negacyclic descaling
uint64_t* get_psi_inv_powers(size_t N, int device_id = 0);

// Cleanup twiddle factors
void cleanup_twiddle_factors();

} // namespace ntt_twiddles

