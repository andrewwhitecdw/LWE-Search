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
#include "../src/lwe_server.h"
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

std::function<int8_t(int8_t, int8_t, int8_t)> get_function(int operation_index)
{
    switch (operation_index)
    {
    case 0:
        return [](int8_t a, int8_t b, int8_t c) -> int8_t {
            return a + b;
        };
        break;
    case 1:
        return [](int8_t a, int8_t b, int8_t c) -> int8_t {
            return a - b;
        };
        break;
    case 2:
        return [](int8_t a, int8_t b, int8_t c) -> int8_t {
            return -b;
        };
        break;
    case 3:
        return [](int8_t a, int8_t b, int8_t c) -> int8_t {
            return b * c;
        };
        break;
    default:
        throw std::invalid_argument("Invalid operation index");
    }
}

int main(int argc, char **argv)
{
    size_t lwe_degree = static_cast<size_t>(argc > 1 ? atoi(argv[1]) : 1536);
    size_t ptxt_len = static_cast<size_t>(argc > 2 ? atoi(argv[2]) : 64);
    size_t batch_size = static_cast<size_t>(argc > 3 ? atoi(argv[3]) : 4);
    int operation_index = static_cast<int>(argc > 4 ? atoi(argv[4]) : 0);

    auto lwe = lwe_ann::create_lwe_context<int32_t>(lwe_degree, 3.2);

    lwe_ann::LWESecretPack<int32_t> secret;
    lwe->generate_secret_pack(secret, batch_size);

    lwe_ann::LWECiphertextPack<int32_t> ctxt1, ctxt2;

    // Generate random input
    std::vector<int8_t> input1(ptxt_len * batch_size);
    std::vector<int8_t> input2(ptxt_len * batch_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8_t> dist(-128, 127);
    for (size_t i = 0; i < ptxt_len * batch_size; i++)
    {
        input1[i] = dist(gen);
        input2[i] = dist(gen);
    }

    int8_t random_constant = dist(gen);

    int32_t scale = 1 << 24; // (maximum for int32_t/int8_t)

    std::vector<int8_t> output(ptxt_len * batch_size);
    lwe->encrypt(ctxt1, input1, scale, secret, batch_size);
    std::cout << "ctxt1.a_data_.size(): " << ctxt1.a_data_.size() << std::endl;
    std::cout << "ctxt1.b_data_.size(): " << ctxt1.b_data_.size() << std::endl;
    std::cout << "ctxt1.scale_: " << ctxt1.scale_ << std::endl;

    lwe->encrypt(ctxt2, input2, scale, secret, batch_size);

    double t0 = gettime();

    auto ctxt1_ld = lwe->ciphertext_to_ld(std::move(ctxt1));
    auto ctxt2_ld = lwe->ciphertext_to_ld(std::move(ctxt2));

    switch (operation_index)
    {
    case 0:
        lwe->add(ctxt1_ld, ctxt1_ld, ctxt2_ld);
        break;
    case 1:
        lwe->sub(ctxt1_ld, ctxt1_ld, ctxt2_ld);
        break;
    case 2:
        lwe->neg(ctxt1_ld, ctxt2_ld);
        break;
    case 3:
        lwe->mult_constant(ctxt1_ld, ctxt2_ld, random_constant);
        break;
    default:
        throw std::invalid_argument("Invalid operation index");
    }

    auto function = get_function(operation_index);

    double t1 = gettime();

    // __ld_to_ciphertext will synchronize the tasks
    auto ctxt_res = lwe->__ld_to_ciphertext(ctxt1_ld);

    double t2 = gettime();

    lwe->decrypt(output, ctxt_res, secret);

    bool testPassed = true;
    for (size_t i = 0; i < ptxt_len * batch_size; i++)
    {
        int8_t ground_truth = function(input1[i], input2[i], random_constant);
        if (output[i] != ground_truth)
        {
            cout << "Error at index " << i << ": " << static_cast<int>(output[i])
                 << " != " << static_cast<int>(ground_truth) << endl;
            testPassed = false;
        }
    }

    double submission_time = t1 - t0;
    double execution_time = t2 - t1;
    std::string testName = extractTestName(argv[0]);

    printTestResults(testName, testPassed, submission_time, execution_time);

    return !testPassed;
}
