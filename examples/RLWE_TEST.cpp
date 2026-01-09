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

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include "../src/rlwe_server.h"
#include "examples.h"

using namespace std;
using namespace seal;
using namespace std::chrono;

/* wall-clock time */
double gettime()
{
    auto now = system_clock::now().time_since_epoch();
    return duration_cast<duration<double>>(now).count();
}

std::function<int32_t(int32_t, int32_t, int32_t)> get_function(int operation_index, int32_t ptxt_modulus)
{
    int32_t half_ptxt_modulus = ptxt_modulus / 2;
    switch (operation_index)
    {
    case 0:
        return [half_ptxt_modulus, ptxt_modulus](int32_t a, int32_t b, int32_t c) -> int32_t {
            int32_t res = a + b;
            if (res >= half_ptxt_modulus)
            {
                res -= ptxt_modulus;
            }
            if (res < -half_ptxt_modulus)
            {
                res += ptxt_modulus;
            }
            return res;
        };
        break;
    case 1:
        return [half_ptxt_modulus, ptxt_modulus](int32_t a, int32_t b, int32_t c) -> int32_t {
            int32_t res = a - b;
            if (res >= half_ptxt_modulus)
            {
                res -= ptxt_modulus;
            }
            if (res < -half_ptxt_modulus)
            {
                res += ptxt_modulus;
            }
            return res;
        };
        break;
    case 2:
        return [half_ptxt_modulus, ptxt_modulus](int32_t a, int32_t b, int32_t c) -> int32_t {
            int32_t res = -b;
            if (res == half_ptxt_modulus) res = -res;
            return res;
        };
        break;
    default:
        throw std::invalid_argument("Invalid operation index");
    }
}

int main(int argc, char **argv)
{
    size_t rlwe_degree = static_cast<size_t>(argc > 1 ? atoi(argv[1]) : 2048);
    size_t batch_size = static_cast<size_t>(argc > 2 ? atoi(argv[2]) : 16);
    int operation_index = static_cast<int>(argc > 3 ? atoi(argv[3]) : 0);

    // This is just arbitrary.
    int32_t ptxt_modulus = 1 << 24;
    uint64_t modulus = 0x7FFFFFE060001ULL; // Hard-coded prime in SEAL
    // uint64_t modulus = ((1ULL << 56) - (1ULL << 32) + 1);

    uint64_t w_scale = modulus / static_cast<uint64_t>(ptxt_modulus);

    // Currently, only supports uint64_t
    auto rlwe = lwe_ann::create_rlwe_context<uint64_t>(rlwe_degree, modulus, 3.2);

    lwe_ann::RLWESecret<uint64_t> secret;
    rlwe->generate_secret(secret);

    lwe_ann::RLWECiphertextPack<uint64_t> ctxt1, ctxt2;

    // Generate random input
    lwe_ann::PolynomialPack<int32_t> input1(rlwe_degree * batch_size);
    lwe_ann::PolynomialPack<int32_t> input2(rlwe_degree * batch_size);
    input1.batch_size_ = batch_size;
    input2.batch_size_ = batch_size;
    input1.is_ntt_ = false;
    input2.is_ntt_ = false;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(-ptxt_modulus / 2, ptxt_modulus / 2 - 1);

    // Corner cases
    input1[0] = -ptxt_modulus / 2;
    input1[1] = 0;
    input1[2] = ptxt_modulus / 2 - 1;
    input1[3] = -ptxt_modulus / 2;
    input1[4] = 0;
    input1[5] = ptxt_modulus / 2 - 1;
    input1[6] = -ptxt_modulus / 2;
    input1[7] = 0;
    input1[8] = ptxt_modulus / 2 - 1;

    input2[0] = -ptxt_modulus / 2;
    input2[1] = -ptxt_modulus / 2;
    input2[2] = -ptxt_modulus / 2;
    input2[3] = 0;
    input2[4] = 0;
    input2[5] = 0;
    input2[6] = ptxt_modulus / 2 - 1;
    input2[7] = ptxt_modulus / 2 - 1;
    input2[8] = ptxt_modulus / 2 - 1;

    for (size_t i = 9; i < rlwe_degree * batch_size; i++)
    {
        input1[i] = dist(gen);
        input2[i] = dist(gen);
    }

    int32_t random_constant = dist(gen);
    std::cout << "random_constant: " << static_cast<int>(random_constant) << std::endl;

    lwe_ann::PolynomialPack<int32_t> output;

    rlwe->encrypt(ctxt1, input1, static_cast<double>(w_scale), secret);

    std::cout << "ctxt1.a_data_.size(): " << ctxt1.a_data_.size() << std::endl;
    std::cout << "ctxt1.b_data_.size(): " << ctxt1.b_data_.size() << std::endl;
    std::cout << "ctxt1.scale_: " << ctxt1.scale_ << std::endl;

    rlwe->encrypt(ctxt2, input2, static_cast<double>(w_scale), secret);

    double t0 = gettime();

    auto ctxt1_ld = rlwe->ciphertext_to_ld(std::move(ctxt1));
    auto ctxt2_ld = rlwe->ciphertext_to_ld(std::move(ctxt2));

    switch (operation_index)
    {
    case 0:
        rlwe->add(ctxt1_ld, ctxt1_ld, ctxt2_ld);
        break;
    case 1:
        rlwe->sub(ctxt1_ld, ctxt1_ld, ctxt2_ld);
        break;
    case 2:
        rlwe->neg(ctxt1_ld, ctxt2_ld);
        break;
    default:
        throw std::invalid_argument("Invalid operation index");
    }

    auto function = get_function(operation_index, ptxt_modulus);
    double t1 = gettime();

    // __ld_to_ciphertext will synchronize the tasks
    auto ctxt_res = rlwe->__ld_to_ciphertext(ctxt1_ld);

    double t2 = gettime();
    rlwe->decrypt(output, ctxt_res, secret);

    std::cout << "ctxt_res.a_data_.size(): " << ctxt_res.a_data_.size() << std::endl;
    std::cout << "ctxt_res.b_data_.size(): " << ctxt_res.b_data_.size() << std::endl;
    std::cout << "ctxt_res.a_data_.is_ntt_: " << ctxt_res.a_data_.is_ntt_ << std::endl;
    std::cout << "ctxt_res.b_data_.is_ntt_: " << ctxt_res.b_data_.is_ntt_ << std::endl;
    std::cout << "ctxt_res.scale_: " << ctxt_res.scale_ << std::endl;
    std::cout << "ctxt_res.round_shift_bits_: " << ctxt_res.round_shift_bits_ << std::endl;

    std::cout << "output.size(): " << output.size() << std::endl;
    std::cout << "output.batch_size_: " << output.batch_size_ << std::endl;
    std::cout << "output.is_ntt_: " << output.is_ntt_ << std::endl;

    bool testPassed = true;
    for (size_t i = 0; i < rlwe_degree * batch_size; i++)
    {
        int32_t ground_truth = function(input1[i], input2[i], random_constant);
        // int32_t ground_truth = input1[i];
        if (output[i] != ground_truth)
        {
            std::cout << "Error at index " << i << ": " << static_cast<int>(output[i])
                 << " != " << static_cast<int>(ground_truth) << std::endl;
            testPassed = false;
        }
    }

    double submission_time = t1 - t0;
    double execution_time = t2 - t1;
    std::string testName = extractTestName(argv[0]);

    printTestResults(testName, testPassed, submission_time, execution_time);
    
    return !testPassed;
}
