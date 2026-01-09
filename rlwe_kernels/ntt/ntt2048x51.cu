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

#include <stdint.h>

// Define uint128_t for 128-bit arithmetic
typedef unsigned __int128 uint128_t;

__global__ void ker_code0_2048(uint64_t  *X, uint64_t  *Y, uint64_t modulus, uint64_t  *twiddles, uint64_t mu) {
    uint128_t q1, q10, q11, q12, q13, q14, q15, q16, 
            q17, q18, q19, q2, q20, q21, q22, q23, 
            q24, q25, q26, q27, q28, q29, q3, q30, 
            q31, q32, q33, q34, q35, q36, q37, q38, 
            q39, q4, q40, q41, q42, q43, q44, q5, 
            q6, q7, q8, q9;
    struct ShmemLayout
    {
        uint64_t T[2 * 2048];
    };
    extern __shared__ __align__(alignof(ShmemLayout)) char smem[];
    auto T1 = reinterpret_cast<uint64_t*>(smem);
    auto T2 = T1 + 2048;

    int a16, a17;
    uint64_t s10, s9;
    a16 = ((2048*blockIdx.x) + threadIdx.x);
    s9 = X[a16];
    q12 = (((uint128_t ) twiddles[1])*((uint128_t ) X[(a16 + 1024)]));  
    q13 = ((((uint128_t ) mu)*(q12 >> 49)) >> 56);
    q14 = (q12 - (q13*((uint128_t ) modulus)));
    s10 = ((uint64_t ) ((((q14 > ((uint128_t ) modulus)))) ? ((q14 - ((uint128_t ) modulus))) : (q14)));
    a17 = (2*threadIdx.x);
    q1 = (((uint128_t ) s9) + ((uint128_t ) s10));
    T2[a17] = ((((q1 > ((uint128_t ) modulus)))) ? ((q1 - ((uint128_t ) modulus))) : (q1));
    T2[(a17 + 1)] = ((((uint128_t ) s9) - ((uint128_t ) s10)) + ((((s9 < s10))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a34, a35;
    uint64_t s19, s20;
    s19 = T2[threadIdx.x];
    a34 = (threadIdx.x + 1024);
    q15 = (((uint128_t ) twiddles[(2 + (a34 % 2))])*((uint128_t ) T2[a34]));  
    q16 = ((((uint128_t ) mu)*(q15 >> 49)) >> 56);
    q17 = (q15 - (q16*((uint128_t ) modulus)));
    s20 = ((uint64_t ) ((((q17 > ((uint128_t ) modulus)))) ? ((q17 - ((uint128_t ) modulus))) : (q17)));
    a35 = (2*threadIdx.x);
    q2 = (((uint128_t ) s19) + ((uint128_t ) s20));
    T1[a35] = ((((q2 > ((uint128_t ) modulus)))) ? ((q2 - ((uint128_t ) modulus))) : (q2));
    T1[(a35 + 1)] = ((((uint128_t ) s19) - ((uint128_t ) s20)) + ((((s19 < s20))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a52, a53;
    uint64_t s29, s30;
    s29 = T1[threadIdx.x];
    a52 = (threadIdx.x + 1024);
    q18 = (((uint128_t ) twiddles[(4 + (a52 % 4))])*((uint128_t ) T1[a52]));  
    q19 = ((((uint128_t ) mu)*(q18 >> 49)) >> 56);
    q20 = (q18 - (q19*((uint128_t ) modulus)));
    s30 = ((uint64_t ) ((((q20 > ((uint128_t ) modulus)))) ? ((q20 - ((uint128_t ) modulus))) : (q20)));
    a53 = (2*threadIdx.x);
    q3 = (((uint128_t ) s29) + ((uint128_t ) s30));
    T2[a53] = ((((q3 > ((uint128_t ) modulus)))) ? ((q3 - ((uint128_t ) modulus))) : (q3));
    T2[(a53 + 1)] = ((((uint128_t ) s29) - ((uint128_t ) s30)) + ((((s29 < s30))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a70, a71;
    uint64_t s39, s40;
    s39 = T2[threadIdx.x];
    a70 = (threadIdx.x + 1024);
    q21 = (((uint128_t ) twiddles[(8 + (a70 % 8))])*((uint128_t ) T2[a70]));  
    q22 = ((((uint128_t ) mu)*(q21 >> 49)) >> 56);
    q23 = (q21 - (q22*((uint128_t ) modulus)));
    s40 = ((uint64_t ) ((((q23 > ((uint128_t ) modulus)))) ? ((q23 - ((uint128_t ) modulus))) : (q23)));
    a71 = (2*threadIdx.x);
    q4 = (((uint128_t ) s39) + ((uint128_t ) s40));
    T1[a71] = ((((q4 > ((uint128_t ) modulus)))) ? ((q4 - ((uint128_t ) modulus))) : (q4));
    T1[(a71 + 1)] = ((((uint128_t ) s39) - ((uint128_t ) s40)) + ((((s39 < s40))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a88, a89;
    uint64_t s49, s50;
    s49 = T1[threadIdx.x];
    a88 = (threadIdx.x + 1024);
    q24 = (((uint128_t ) twiddles[(16 + (a88 % 16))])*((uint128_t ) T1[a88]));  
    q25 = ((((uint128_t ) mu)*(q24 >> 49)) >> 56);
    q26 = (q24 - (q25*((uint128_t ) modulus)));
    s50 = ((uint64_t ) ((((q26 > ((uint128_t ) modulus)))) ? ((q26 - ((uint128_t ) modulus))) : (q26)));
    a89 = (2*threadIdx.x);
    q5 = (((uint128_t ) s49) + ((uint128_t ) s50));
    T2[a89] = ((((q5 > ((uint128_t ) modulus)))) ? ((q5 - ((uint128_t ) modulus))) : (q5));
    T2[(a89 + 1)] = ((((uint128_t ) s49) - ((uint128_t ) s50)) + ((((s49 < s50))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a106, a107;
    uint64_t s59, s60;
    s59 = T2[threadIdx.x];
    a106 = (threadIdx.x + 1024);
    q27 = (((uint128_t ) twiddles[(32 + (a106 % 32))])*((uint128_t ) T2[a106]));  
    q28 = ((((uint128_t ) mu)*(q27 >> 49)) >> 56);
    q29 = (q27 - (q28*((uint128_t ) modulus)));
    s60 = ((uint64_t ) ((((q29 > ((uint128_t ) modulus)))) ? ((q29 - ((uint128_t ) modulus))) : (q29)));
    a107 = (2*threadIdx.x);
    q6 = (((uint128_t ) s59) + ((uint128_t ) s60));
    T1[a107] = ((((q6 > ((uint128_t ) modulus)))) ? ((q6 - ((uint128_t ) modulus))) : (q6));
    T1[(a107 + 1)] = ((((uint128_t ) s59) - ((uint128_t ) s60)) + ((((s59 < s60))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a124, a125;
    uint64_t s69, s70;
    s69 = T1[threadIdx.x];
    a124 = (threadIdx.x + 1024);
    q30 = (((uint128_t ) twiddles[(64 + (a124 % 64))])*((uint128_t ) T1[a124]));  
    q31 = ((((uint128_t ) mu)*(q30 >> 49)) >> 56);
    q32 = (q30 - (q31*((uint128_t ) modulus)));
    s70 = ((uint64_t ) ((((q32 > ((uint128_t ) modulus)))) ? ((q32 - ((uint128_t ) modulus))) : (q32)));
    a125 = (2*threadIdx.x);
    q7 = (((uint128_t ) s69) + ((uint128_t ) s70));
    T2[a125] = ((((q7 > ((uint128_t ) modulus)))) ? ((q7 - ((uint128_t ) modulus))) : (q7));
    T2[(a125 + 1)] = ((((uint128_t ) s69) - ((uint128_t ) s70)) + ((((s69 < s70))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a142, a143;
    uint64_t s79, s80;
    s79 = T2[threadIdx.x];
    a142 = (threadIdx.x + 1024);
    q33 = (((uint128_t ) twiddles[(128 + (a142 % 128))])*((uint128_t ) T2[a142]));  
    q34 = ((((uint128_t ) mu)*(q33 >> 49)) >> 56);
    q35 = (q33 - (q34*((uint128_t ) modulus)));
    s80 = ((uint64_t ) ((((q35 > ((uint128_t ) modulus)))) ? ((q35 - ((uint128_t ) modulus))) : (q35)));
    a143 = (2*threadIdx.x);
    q8 = (((uint128_t ) s79) + ((uint128_t ) s80));
    T1[a143] = ((((q8 > ((uint128_t ) modulus)))) ? ((q8 - ((uint128_t ) modulus))) : (q8));
    T1[(a143 + 1)] = ((((uint128_t ) s79) - ((uint128_t ) s80)) + ((((s79 < s80))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a160, a161;
    uint64_t s89, s90;
    s89 = T1[threadIdx.x];
    a160 = (threadIdx.x + 1024);
    q36 = (((uint128_t ) twiddles[(256 + (a160 % 256))])*((uint128_t ) T1[a160]));  
    q37 = ((((uint128_t ) mu)*(q36 >> 49)) >> 56);
    q38 = (q36 - (q37*((uint128_t ) modulus)));
    s90 = ((uint64_t ) ((((q38 > ((uint128_t ) modulus)))) ? ((q38 - ((uint128_t ) modulus))) : (q38)));
    a161 = (2*threadIdx.x);
    q9 = (((uint128_t ) s89) + ((uint128_t ) s90));
    T2[a161] = ((((q9 > ((uint128_t ) modulus)))) ? ((q9 - ((uint128_t ) modulus))) : (q9));
    T2[(a161 + 1)] = ((((uint128_t ) s89) - ((uint128_t ) s90)) + ((((s89 < s90))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a178, a179;
    uint64_t s100, s99;
    s99 = T2[threadIdx.x];
    a178 = (threadIdx.x + 1024);
    q39 = (((uint128_t ) twiddles[(512 + (a178 % 512))])*((uint128_t ) T2[a178]));  
    q40 = ((((uint128_t ) mu)*(q39 >> 49)) >> 56);
    q41 = (q39 - (q40*((uint128_t ) modulus)));
    s100 = ((uint64_t ) ((((q41 > ((uint128_t ) modulus)))) ? ((q41 - ((uint128_t ) modulus))) : (q41)));
    a179 = (2*threadIdx.x);
    q10 = (((uint128_t ) s99) + ((uint128_t ) s100));
    T1[a179] = ((((q10 > ((uint128_t ) modulus)))) ? ((q10 - ((uint128_t ) modulus))) : (q10));
    T1[(a179 + 1)] = ((((uint128_t ) s99) - ((uint128_t ) s100)) + ((((s99 < s100))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
    int a201, a202;
    uint64_t s109, s110;
    s109 = T1[threadIdx.x];
    a201 = (threadIdx.x + 1024);
    q42 = (((uint128_t ) twiddles[(1024 + (a201 % 1024))])*((uint128_t ) T1[a201]));  
    q43 = ((((uint128_t ) mu)*(q42 >> 49)) >> 56);
    q44 = (q42 - (q43*((uint128_t ) modulus)));
    s110 = ((uint64_t ) ((((q44 > ((uint128_t ) modulus)))) ? ((q44 - ((uint128_t ) modulus))) : (q44)));
    a202 = ((2048*blockIdx.x) + (2*threadIdx.x));
    q11 = (((uint128_t ) s109) + ((uint128_t ) s110));
    Y[a202] = ((((q11 > ((uint128_t ) modulus)))) ? ((q11 - ((uint128_t ) modulus))) : (q11));
    Y[(a202 + 1)] = ((((uint128_t ) s109) - ((uint128_t ) s110)) + ((((s109 < s110))) ? (((uint128_t ) modulus)) : (0)));
    __syncthreads();
}
