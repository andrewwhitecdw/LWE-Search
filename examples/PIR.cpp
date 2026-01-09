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

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include "../src/reconstruct_words.h"
#include "../src/simplepir.h"
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

std::vector<double> generate_random_normalized_vector(size_t dim)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> dist(0, 1);
    std::vector<double> vec(dim);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
    double norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    std::transform(vec.begin(), vec.end(), vec.begin(), [&](double x) { return x / norm; });
    return vec;
}

template <typename T>
void print_matrix(T *matrix, size_t M, size_t N, bool row_major = true)
{
    std::cout << "[" << std::endl;
    for (size_t i = 0; i < M; i++)
    {
        std::cout << "[";
        for (size_t j = 0; j < N; j++)
        {
            auto value = row_major ? matrix[i * N + j] : matrix[j * M + i];
            std::cout << static_cast<int>(value);
            if (j != N - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i != M - 1)
        {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    // Parameter explanation:
    // 1. lwe_degree: the degree of the LWE encryption
    // 2. dim_rest: the last dimension of the database (= number of records per cluster)
    // 3. num_clusters: the number of clusters, PIR attempts to retrieve one of the clusters
    // 4. vector_len: the dimension of each vector embedding. If 1, it becomes a plain PIR.
    // 5. batch_size: the number of queries to be processed together,
    // 6. lwe_round_shift_bits: the number of bits to discard from each element of the LWE ciphertexts after the first
    //                          GEMM.
    // 7. rlwe_round_shift_bits: the number of bits to discard from each element of the RLWE ciphertexts after
    //                           LWE-to-RLWE packing (response compression).
    // 8. use_rlwe_packing: whether to use LWE-to-RLWE packing (response compression).
    size_t lwe_degree = static_cast<size_t>(argc > 1 ? atoi(argv[1]) : 1536);
    size_t dim_rest = static_cast<size_t>(argc > 2 ? atoi(argv[2]) : 8192); // 8192
    size_t num_clusters = static_cast<size_t>(argc > 3 ? atoi(argv[3]) : 4096); // 2250
    size_t vector_len = static_cast<size_t>(argc > 4 ? atoi(argv[4]) : 96); // 96
    size_t batch_size = static_cast<size_t>(argc > 5 ? atoi(argv[5]) : 16);
    size_t lwe_round_shift_bits = static_cast<size_t>(argc > 6 ? atoi(argv[6]) : 16);
    size_t rlwe_round_shift_bits = static_cast<size_t>(argc > 7 ? atoi(argv[7]) : 30);
    bool use_rlwe_packing = static_cast<bool>(argc > 8 ? atoi(argv[8]) > 0 : true);

    size_t rlwe_degree = 2048;
    uint64_t rlwe_modulus = 0x7FFFFFE060001ULL;

    std::vector<int> device_ids = { 0 };
    size_t num_gpus = device_ids.size();

    auto lwe = lwe_ann::create_lwe_context<int32_t>(lwe_degree, 3.2);
    auto &ctx = lwe->cudastf_ctx_;
    lwe_ann::RLWEContextPtr<uint64_t> rlwe = nullptr;
    if (use_rlwe_packing)
    {
        rlwe = lwe_ann::create_rlwe_context<uint64_t>(rlwe_degree, rlwe_modulus, 3.2, ctx);
    }
    else
    {
        rlwe_round_shift_bits = 0;
    }

    // Make a dummy database
    std::cout << "Generating dummy database... (this may take a while)" << std::endl;
    size_t dim_first = vector_len * num_clusters;

    // This is determined by the tile_k_max_ in SimplePIRServer
    size_t padded_dim_first = lwe_ann::pad_by<size_t>(dim_first, 32);
    std::vector<int8_t> db(dim_rest * dim_first);

    int32_t plaintext_scale = 127;

    for (auto it = db.begin(); it != db.end(); it += vector_len)
    {
        auto vec = generate_random_normalized_vector(vector_len);
        std::transform(
            vec.begin(), vec.end(), it, [&](double val) { return static_cast<int8_t>(val * plaintext_scale); });
    }
    std::cout << "Done!" << std::endl;

    // Make a dummy query (just assuming that the index is known for now)
    std::uniform_int_distribution<size_t> index_dist(0, num_clusters - 1);
    std::vector<int8_t> query(padded_dim_first * batch_size, 0);
    std::vector<size_t> indices(batch_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t b = 0; b < batch_size; b++)
    {
        size_t index = index_dist(gen);
        indices[b] = index;
        std::cout << "Selected index: " << index << std::endl;
        auto vec = generate_random_normalized_vector(vector_len);
        std::transform(
            vec.begin(), vec.end(), query.begin() + b * padded_dim_first + index * vector_len,
            [&](double val) { return static_cast<int8_t>(val * plaintext_scale); });
    }

    // Ground truth result
    std::vector<int32_t> result(dim_rest * batch_size, 0);

    int32_t half_plaintext_scale = plaintext_scale >> 1;

    for (size_t b = 0; b < batch_size; b++)
    {
        size_t index = indices[b];
        for (size_t i = 0; i < dim_rest; i++)
        {
            int32_t sum = 0;
            for (size_t j = 0; j < vector_len; j++)
            {
                int32_t db_val = static_cast<int32_t>(db[i * dim_first + index * vector_len + j]);
                int32_t query_val = static_cast<int32_t>(query[b * padded_dim_first + index * vector_len + j]);
                sum += db_val * query_val;
            }
            // Round to nearest
            sum = sum < 0 ? sum - half_plaintext_scale : sum + half_plaintext_scale;
            sum /= plaintext_scale;
            result[b * dim_rest + i] = sum;
        }
    }

    // Server-side offline setup
    std::cout << "DB size: " << db.size() / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Preparing PIR server..." << std::endl;
    lwe_ann::SimplePIRServer<int32_t, uint64_t, int8_t> pir(
        std::move(db), std::vector<size_t>{ dim_first, dim_rest }, lwe, lwe_round_shift_bits, rlwe,
        rlwe_round_shift_bits, device_ids, batch_size);

    if (padded_dim_first != pir.get_padded_dim(0))
    {
        throw std::runtime_error("padded_dim_first != pir.get_padded_dim(0)");
    }
    size_t padded_dim_rest = pir.get_padded_dim(1) * pir.get_padded_dim(2);

    pir.offline_setup();
    std::cout << "Done!" << std::endl;

    // *** You should not access db after this point ***

    // Client-side offline setup
    lwe_ann::LWESecretPack<int32_t> lwe_secret;
    lwe_ann::LWEClient<int32_t> lwe_client(lwe_degree, 3.2);
    lwe_ann::RLWEClient<uint64_t> rlwe_client(rlwe_degree, rlwe_modulus, 3.2, lwe_client.prng_factory_);

    int32_t ctxt_scale = 1 << 17;

    std::vector<int32_t> a_data_client, b_data_query;
#ifdef DEBUG_ZERO_A
    a_data_client.resize(padded_dim_first * lwe_degree);
    std::fill(a_data_client.begin(), a_data_client.end(), 0);
#else
    lwe_client.sample_a_from_seed(a_data_client, padded_dim_first * lwe_degree, pir.get_a_seed(0));
#endif
    lwe_ann::RLWESecret<uint64_t> rlwe_secret;
    rlwe_client.generate_secret(rlwe_secret);

    // Online stage
    // 1. client encryption (and query communication)
    lwe_client.generate_secret_pack(lwe_secret, batch_size);
    lwe_client.compute_b_from_a(b_data_query, a_data_client, query, ctxt_scale, lwe_secret, batch_size);

    // Just assume that the data is pinned
    std::cout << "Pinning query..." << std::endl;
    std::vector<int32_t *> query_pinned(1);
    cudastf::cuda_safe_call(cudaMallocHost(&query_pinned[0], b_data_query.size() * sizeof(int32_t)));
    memcpy(query_pinned[0], b_data_query.data(), b_data_query.size() * sizeof(int32_t));
    std::cout << "Done!" << std::endl;

    // Conversion ciphertexts"
    std::cout << "Pinning conversion ciphertexts..." << std::endl;
    std::vector<uint64_t *> conversion_ctxts_pinned;
    if (use_rlwe_packing)
    {
        lwe_ann::RLWECiphertextPack<uint64_t> conversion_ctxts;
        size_t lwe_bits = sizeof(int32_t) * 8 - lwe_round_shift_bits;
        uint64_t rlwe_scale = rlwe_modulus / (1 << lwe_bits);
        rlwe_client.generate_secret(rlwe_secret, true);
        rlwe_client.prepare_lwe_to_rlwe_conversion_ciphertexts(
            conversion_ctxts, lwe_secret, lwe_degree, rlwe_scale, rlwe_secret, false);
        conversion_ctxts_pinned.resize(2);
        cudastf::cuda_safe_call(
            cudaMallocHost(&conversion_ctxts_pinned[0], conversion_ctxts.a_data_.size() * sizeof(uint64_t)));
        cudastf::cuda_safe_call(
            cudaMallocHost(&conversion_ctxts_pinned[1], conversion_ctxts.b_data_.size() * sizeof(uint64_t)));
        memcpy(
            conversion_ctxts_pinned[0], conversion_ctxts.a_data_.data(),
            conversion_ctxts.a_data_.size() * sizeof(uint64_t));
        memcpy(
            conversion_ctxts_pinned[1], conversion_ctxts.b_data_.data(),
            conversion_ctxts.b_data_.size() * sizeof(uint64_t));
    }
    std::cout << "Done!" << std::endl;

    // Warm-up run
    std::vector<std::vector<int8_t>> pir_res =
        pir.online_stage_synchronized(query_pinned, batch_size, conversion_ctxts_pinned);

    for (auto devid : device_ids)
    {
        cudaSetDevice(devid);
        cudaStreamSynchronize(ctx->fence());
    }
    std::cout << "Doing actual run..." << std::endl;
    // We are only interested in the time spent on the server-side
    double t0 = gettime();

    size_t iters = 10;
    for (size_t i = 0; i < iters; i++)
    {
        cudastf::nvtx_range r("LWE_ANN_Online");
        auto _not_used = pir.online_stage_synchronized(query_pinned, batch_size, conversion_ctxts_pinned);
    }

    double t1 = gettime();
    // Synchronization has no meaning here
    for (auto devid : device_ids)
    {
        cudaSetDevice(devid);
        cudaStreamSynchronize(ctx->fence());
    }
    double t2 = gettime();

    // 3. client reconstruction and decryption
    std::vector<int8_t> final_res;

    if (use_rlwe_packing)
    {
        size_t num_chunks = pir.get_num_rlwe_plain_word_chunks();
        size_t lwe_bits = sizeof(int32_t) * 8 - lwe_round_shift_bits;
        uint64_t rlwe_scale = rlwe_modulus / (1 << lwe_bits);
        uint64_t decrypt_scale = rlwe_scale * ((plaintext_scale * ctxt_scale) >> lwe_round_shift_bits);
        for (size_t b = 0; b < batch_size; b++)
        {
            auto tmp = lwe_ann::decrypt_pir_res_rlwe_2d<uint64_t, int8_t>(
                rlwe_client, rlwe_secret, pir_res[b], rlwe_degree, device_ids.size(), dim_rest, pir.get_padded_dim(2),
                num_chunks, decrypt_scale, rlwe_round_shift_bits);
            final_res.insert(final_res.end(), tmp.begin(), tmp.end());
        }
    }
    else
    {
        size_t num_chunks = pir.get_num_lwe_plain_word_chunks();
        std::vector<int32_t> pir_res_packed;

        for (size_t i = 0; i < batch_size; i++)
        {
            auto tmp = lwe_ann::reconstruct_from_plain_words_lwe<int32_t, int8_t>(pir_res[i], num_chunks);
            pir_res_packed.insert(pir_res_packed.end(), tmp.begin(), tmp.end());
        }
        lwe_ann::LWECiphertextPack<int32_t> ctxt;
        ctxt.a_data_ = pir.hint_gathered_;
        ctxt.b_data_ = pir_res_packed;
        ctxt.scale_ = plaintext_scale * ctxt_scale;
        ctxt.round_shift_bits_ = lwe_round_shift_bits;

        std::vector<int8_t> decrypted_res;
        lwe_client.decrypt(decrypted_res, ctxt, lwe_secret);
        auto tmp = lwe_ann::reorganize_2d_without_packing(decrypted_res, dim_rest, pir.get_padded_dim(2), batch_size);
        final_res.insert(final_res.end(), tmp.begin(), tmp.end());
    }

    bool testPassed = true;
    int error_cnt = 0;
    for (size_t i = 0; i < dim_rest * batch_size; i++)
    {
        int32_t pir_res_val = final_res[i];
        int32_t true_val = result[i];
        int32_t diff = pir_res_val - true_val;
        if (diff < -1 || diff > 1)
        {
            if (error_cnt < 100)
            {
                cout << "Error at index " << i << ": " << pir_res_val << " != " << true_val << endl;
            }
            if (error_cnt == 100)
            {
                cout << "..." << endl;
            }
            testPassed = false;
            error_cnt++;
        }
    }
    std::cout << "Error count: " << error_cnt << std::endl;

    double submission_time = t1 - t0;
    double execution_time = t2 - t1;
    std::string testName = extractTestName(argv[0]);

    printTestResults(testName, testPassed, submission_time, execution_time);

    for (auto &ptr : query_pinned)
    {
        cudastf::cuda_safe_call(cudaFreeHost(ptr));
    }
    for (auto &ptr : conversion_ctxts_pinned)
    {
        cudastf::cuda_safe_call(cudaFreeHost(ptr));
    }

    return !testPassed;
}
