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

#include "ntt_twiddles.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace ntt_twiddles {

namespace {
    std::map<int, uint64_t*> device_twd_2048;
    std::map<int, uint64_t*> device_twd_inv_2048;
    
    // Psi powers for negacyclic scaling
    std::map<int, uint64_t*> device_psi_powers_2048;
    std::map<int, uint64_t*> device_psi_inv_powers_2048;
    
    bool initialized = false;
    uint64_t current_modulus = 0;
    uint64_t inv_2048 = 0;  // 2048^{-1} mod q
    uint64_t psi_2048 = 0;  // 2N-th primitive root
    uint64_t psi_inv_2048 = 0;
}

// Bit-reverse an array
static void bit_reverse_array(uint64_t* arr, size_t length) {
    if (length <= 1) return;
    size_t bits = 0;
    size_t temp = length;
    while (temp > 1) {
        bits++;
        temp >>= 1;
    }
    for (size_t i = 0; i < length; i++) {
        size_t rev = 0;
        for (size_t b = 0; b < bits; b++) {
            if (i & (1 << b)) {
                rev |= 1 << (bits - 1 - b);
            }
        }
        if (i < rev) {
            uint64_t tmp = arr[i];
            arr[i] = arr[rev];
            arr[rev] = tmp;
        }
    }
}

// Compute x^y % modulus
static uint64_t powMod(uint64_t x, uint64_t y, uint64_t modulus) {
    uint64_t res = 1;
    __uint128_t tmp;
    while (y > 0) {
        if (y & 1) {
            tmp = (__uint128_t)res * x;
            res = tmp % modulus;
        }
        y >>= 1;
        tmp = (__uint128_t)x * x;
        x = tmp % modulus;
    }
    return res;
}

// Compute modular inverse: x^{-1} mod p
static uint64_t mod_inverse(uint64_t x, uint64_t p) {
    if (x == 0 || x >= p) {
        x = x % p;
    }
    if (x == 0) {
        fprintf(stderr, "Error: Cannot compute modular inverse of 0\n");
        abort();
    }
    return powMod(x, p - 2, p);
}

// Find minimal primitive root of given degree
static uint64_t find_minimal_primitive_root(size_t degree, uint64_t modulus) {
    bool special = (modulus == 0x7FFFFFE060001ULL);
    uint64_t old_degree = degree;
    
    if (special) {
        degree = 1 << 17;  // Use 2^17 for special primes
    }
    
    uint64_t exponent = (modulus - 1) / degree;
    uint64_t root = 0;
    
    for (uint64_t generator = 2; generator < 1000; generator++) {
        root = powMod(generator, exponent, modulus);
        uint64_t test_degree = powMod(root, degree, modulus);
        uint64_t test_half = powMod(root, degree / 2, modulus);
        
        if (test_degree == 1 && test_half == (modulus - 1)) {
            break;
        }
    }
    
    // Search for minimal among even powers
    uint64_t generator_sq = (__uint128_t)root * root % modulus;
    uint64_t current_generator = root;
    
    for (size_t i = 0; i < degree; i += 2) {
        if (current_generator < root) {
            root = current_generator;
        }
        __uint128_t tmp = (__uint128_t)current_generator * generator_sq;
        current_generator = tmp % modulus;
    }
    
    // For special primes, exponentiate back to the requested degree
    if (special) {
        root = powMod(root, degree / old_degree, modulus);
    }
    
    return root;
}

void init_twiddle_factors(uint64_t modulus, size_t poly_degree, int device_id) {
    if (initialized && modulus == current_modulus) {
        return;  // Already initialized
    }
   
    if (initialized && modulus != current_modulus) {
        fprintf(stderr, "Error: Attempting to reinitialize with different modulus!\n");
        abort();
    }
   
    current_modulus = modulus;
   
    // Only support N=2048
    constexpr size_t N = 2048;
    
    if (poly_degree != N) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", poly_degree);
        abort();
    }
    
    // Compute N^{-1} mod q
    inv_2048 = mod_inverse(N, modulus);
    
    // Compute 2N-th primitive root (psi) for negacyclic scaling
    psi_2048 = find_minimal_primitive_root(2 * N, modulus);
    psi_inv_2048 = mod_inverse(psi_2048, modulus);
          
    // Compute forward root (psi^2 = N-th primitive root)
    __uint128_t psi_sq_tmp = (__uint128_t)psi_2048 * psi_2048;
    uint64_t forward_root = psi_sq_tmp % modulus;
    uint64_t inverse_root = mod_inverse(forward_root, modulus);
    
    // Generate forward and inverse twiddles (single set, shared across all batches)
    size_t nstages = 11;  // log2(2048)
    
    uint64_t* host_twiddles_fwd = new uint64_t[N];
    uint64_t* host_twiddles_inv = new uint64_t[N];
    
    // Forward twiddles
    size_t idx = 0;
    for (size_t i = 0; i < nstages; i++) {
        size_t stage_size = 1 << i;
        uint64_t* stage_twds = new uint64_t[stage_size];
        
        for (size_t j = 0; j < stage_size; j++) {
            uint64_t exp = (j * N) / (1 << (i + 1));
            stage_twds[j] = powMod(forward_root, exp, modulus);
        }
        
        bit_reverse_array(stage_twds, stage_size);
        
        for (size_t j = 0; j < stage_size; j++) {
            host_twiddles_fwd[idx++] = stage_twds[j];
        }
        
        delete[] stage_twds;
    }
    
    // Prepend 1 at the beginning
    for (size_t i = idx; i > 0; i--) {
        host_twiddles_fwd[i] = host_twiddles_fwd[i-1];
    }
    host_twiddles_fwd[0] = 1;
    
    // Inverse twiddles (same process with inverse root)
    idx = 0;
    for (size_t i = 0; i < nstages; i++) {
        size_t stage_size = 1 << i;
        uint64_t* stage_twds = new uint64_t[stage_size];
        
        for (size_t j = 0; j < stage_size; j++) {
            uint64_t exp = (j * N) / (1 << (i + 1));
            stage_twds[j] = powMod(inverse_root, exp, modulus);
        }
        
        bit_reverse_array(stage_twds, stage_size);
        
        for (size_t j = 0; j < stage_size; j++) {
            host_twiddles_inv[idx++] = stage_twds[j];
        }
        
        delete[] stage_twds;
    }
    
    for (size_t i = idx; i > 0; i--) {
        host_twiddles_inv[i] = host_twiddles_inv[i-1];
    }
    host_twiddles_inv[0] = 1;
    
    // Generate psi powers and inverse powers
    uint64_t* host_psi_powers = new uint64_t[N];
    uint64_t* host_psi_inv_powers = new uint64_t[N];
    
    host_psi_powers[0] = 1;
    host_psi_inv_powers[0] = 1;
    
    for (size_t i = 1; i < N; i++) {
        __uint128_t tmp = (__uint128_t)host_psi_powers[i-1] * psi_2048;
        host_psi_powers[i] = tmp % modulus;
        
        tmp = (__uint128_t)host_psi_inv_powers[i-1] * psi_inv_2048;
        host_psi_inv_powers[i] = tmp % modulus;
    }
    
    // Upload to device
    cudaSetDevice(device_id);
    
    uint64_t* d_twd_fwd;
    uint64_t* d_twd_inv;
    uint64_t* d_psi_powers;
    uint64_t* d_psi_inv_powers;
    
    cudaMalloc(&d_twd_fwd, N * sizeof(uint64_t));
    cudaMalloc(&d_twd_inv, N * sizeof(uint64_t));
    cudaMalloc(&d_psi_powers, N * sizeof(uint64_t));
    cudaMalloc(&d_psi_inv_powers, N * sizeof(uint64_t));
    
    cudaMemcpy(d_twd_fwd, host_twiddles_fwd, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twd_inv, host_twiddles_inv, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_powers, host_psi_powers, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_inv_powers, host_psi_inv_powers, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    device_twd_2048[device_id] = d_twd_fwd;
    device_twd_inv_2048[device_id] = d_twd_inv;
    device_psi_powers_2048[device_id] = d_psi_powers;
    device_psi_inv_powers_2048[device_id] = d_psi_inv_powers;
    
    delete[] host_twiddles_fwd;
    delete[] host_twiddles_inv;
    delete[] host_psi_powers;
    delete[] host_psi_inv_powers;
            
    initialized = true;
}

uint64_t* get_twiddles(size_t size, int device_id, int direction) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
   
    if (size != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", size);
        abort();
    }
   
    bool use_inverse = (direction == 1);
    auto it = use_inverse ? device_twd_inv_2048.find(device_id) : device_twd_2048.find(device_id);
    auto& map = use_inverse ? device_twd_inv_2048 : device_twd_2048;
    
    if (it == map.end()) {
        fprintf(stderr, "Error: Twiddles not found for device %d\n", device_id);
        abort();
    }
    
    return it->second;
}

uint64_t get_inv_N(size_t N) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
    
    if (N != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", N);
        abort();
    }
    
    return inv_2048;
}

uint64_t get_psi(size_t N) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
    
    if (N != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", N);
        abort();
    }
    
    return psi_2048;
}

uint64_t get_psi_inv(size_t N) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
    
    if (N != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", N);
        abort();
    }
    
    return psi_inv_2048;
}

uint64_t* get_psi_powers(size_t N, int device_id) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
    
    if (N != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", N);
        abort();
    }
    
    auto it = device_psi_powers_2048.find(device_id);
    if (it == device_psi_powers_2048.end()) {
        fprintf(stderr, "Error: Psi powers not found for device %d\n", device_id);
        abort();
    }
    
    return it->second;
}

uint64_t* get_psi_inv_powers(size_t N, int device_id) {
    if (!initialized) {
        fprintf(stderr, "Error: Twiddle factors not initialized!\n");
        abort();
    }
    
    if (N != 2048) {
        fprintf(stderr, "Error: Only N=2048 is supported, got %zu\n", N);
        abort();
    }
    
    auto it = device_psi_inv_powers_2048.find(device_id);
    if (it == device_psi_inv_powers_2048.end()) {
        fprintf(stderr, "Error: Psi inverse powers not found for device %d\n", device_id);
        abort();
    }
    
    return it->second;
}

void cleanup_twiddle_factors() {
    for (auto& pair : device_twd_2048) {
        cudaFree(pair.second);
    }
    for (auto& pair : device_twd_inv_2048) {
        cudaFree(pair.second);
    }
    for (auto& pair : device_psi_powers_2048) {
        cudaFree(pair.second);
    }
    for (auto& pair : device_psi_inv_powers_2048) {
        cudaFree(pair.second);
    }
    
    device_twd_2048.clear();
    device_twd_inv_2048.clear();
    device_psi_powers_2048.clear();
    device_psi_inv_powers_2048.clear();
    
    initialized = false;
}

} // namespace ntt_twiddles
