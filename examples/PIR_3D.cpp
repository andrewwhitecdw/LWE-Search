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
    // 2. dim_last: the last dimension of the database (= number of records per cluster)
    // 3. num_clusters_per_meso: the number of clusters per mesocluster.
    // 4. num_meso_clusters: the number of mesoclusters.
    // 5. vector_len: the dimension of each vector embedding. If 1, it becomes a plain PIR.
    // 6. batch_size: the number of queries to be processed together,
    // 7. lwe_round_shift_bits: the number of bits to discard from each element of the LWE ciphertexts after each query
    //                          multiplication (= dimension reduction) step.
    // 8. rlwe_round_shift_bits: the number of bits to discard from each element of the RLWE ciphertexts after each
    //                           LWE-to-RLWE packing (response compression or intermediate result compression) step.
    size_t lwe_degree = static_cast<size_t>(argc > 1 ? atoi(argv[1]) : 1536);
    size_t dim_last = static_cast<size_t>(argc > 2 ? atoi(argv[2]) : 2048); // 8192
    size_t num_clusters_per_meso = static_cast<size_t>(argc > 3 ? atoi(argv[3]) : 128); // 2250
    size_t num_meso_clusters = static_cast<size_t>(argc > 4 ? atoi(argv[4]) : 128);
    size_t vector_len = static_cast<size_t>(argc > 5 ? atoi(argv[5]) : 96); // 96
    size_t batch_size = static_cast<size_t>(argc > 6 ? atoi(argv[6]) : 16);
    size_t lwe_round_shift_bits = static_cast<size_t>(argc > 7 ? atoi(argv[7]) : 16);
    size_t rlwe_round_shift_bits = static_cast<size_t>(argc > 8 ? atoi(argv[8]) : 30);

    size_t rlwe_degree = 2048;
    uint64_t rlwe_modulus = 0x7FFFFFE060001ULL;

    std::vector<int> device_ids = { 0 };
    size_t num_gpus = device_ids.size();

    auto lwe = lwe_ann::create_lwe_context<int32_t>(lwe_degree, 3.2);
    auto &ctx = lwe->cudastf_ctx_;
    lwe_ann::RLWEContextPtr<uint64_t> rlwe =
        lwe_ann::create_rlwe_context<uint64_t>(rlwe_degree, rlwe_modulus, 3.2, ctx);

    // Make a dummy database
    std::cout << "Generating dummy database... (this may take a while)" << std::endl;
    size_t dim_first = vector_len * num_meso_clusters;
    size_t dim_mid = num_clusters_per_meso;

    // This is determined by the tile_k_max_ = 32 in SimplePIRServer
    if (32 % num_gpus != 0)
    {
        throw std::runtime_error("32 % num_gpus != 0");
    }
    size_t padded_dim_first = lwe_ann::pad_by<size_t>(dim_first, 32);
    size_t padded_dim_mid = lwe_ann::pad_by<size_t>(dim_mid, 32);
    std::vector<int8_t> db(dim_first * dim_mid * dim_last);

    int32_t plaintext_scale = 128;
    for (auto it = db.begin(); it != db.end(); it += vector_len)
    {
        auto vec = generate_random_normalized_vector(vector_len);
        std::transform(
            vec.begin(), vec.end(), it, [&](double val) { return static_cast<int8_t>(val * plaintext_scale); });
    }
    std::cout << "Done!" << std::endl;

    // Make a dummy query (just assuming that the index is known for now)
    std::vector<size_t> meso_cluster_indices(batch_size);
    std::vector<size_t> cluster_indices(batch_size);
    std::uniform_int_distribution<size_t> meso_index_dist(0, num_meso_clusters - 1);
    std::uniform_int_distribution<size_t> index_dist(0, num_clusters_per_meso - 1);

    std::vector<int8_t> query_first(padded_dim_first * batch_size, 0);
    std::vector<int8_t> query_second(padded_dim_mid * batch_size, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t b = 0; b < batch_size; b++)
    {
        meso_cluster_indices[b] = meso_index_dist(gen);
        cluster_indices[b] = index_dist(gen);
        std::cout << "Selected index: (" << meso_cluster_indices[b] << ", " << cluster_indices[b] << ")" << std::endl;
        auto vec = generate_random_normalized_vector(vector_len);
        std::transform(
            vec.begin(), vec.end(), query_first.begin() + b * padded_dim_first + meso_cluster_indices[b] * vector_len,
            [&](double val) { return static_cast<int8_t>(val * plaintext_scale); });
        // Second dimension is a one-hot vector
        query_second[b * padded_dim_mid + cluster_indices[b]] = 1;
    }

    // Ground truth result
    std::vector<int32_t> result(dim_last * batch_size, 0);

    int32_t half_plaintext_scale = plaintext_scale >> 1;

    for (size_t b = 0; b < batch_size; b++)
    {
        size_t index = meso_cluster_indices[b] + cluster_indices[b] * num_meso_clusters;
        for (size_t i = 0; i < dim_last; i++)
        {
            int32_t sum = 0;
            for (size_t j = 0; j < vector_len; j++)
            {
                int32_t db_val = static_cast<int32_t>(db[i * dim_first * dim_mid + index * vector_len + j]);
                int32_t query_val =
                    static_cast<int32_t>(query_first[b * padded_dim_first + meso_cluster_indices[b] * vector_len + j]);
                sum += db_val * query_val;
            }
            // Round to nearest
            sum = sum < 0 ? sum - half_plaintext_scale : sum + half_plaintext_scale;
            sum /= plaintext_scale;
            result[b * dim_last + i] = sum;
        }
    }

    // Server-side offline setup
    std::cout << "DB size: " << db.size() / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Preparing PIR server..." << std::endl;
    lwe_ann::SimplePIRServer<int32_t, uint64_t, int8_t> pir(
        std::move(db), std::vector<size_t>{ dim_first, dim_mid, dim_last }, lwe, lwe_round_shift_bits, rlwe,
        rlwe_round_shift_bits, device_ids, batch_size);

    if (padded_dim_first != pir.get_padded_dim(0))
    {
        throw std::runtime_error("padded_dim_first != pir.get_padded_dim(0)");
    }
    if (padded_dim_mid != pir.get_padded_dim(1))
    {
        throw std::runtime_error("padded_dim_mid != pir.get_padded_dim(1)");
    }
    size_t padded_dim_last = pir.get_padded_dim(2);

    pir.offline_setup();
    std::cout << "Offline setup Done!" << std::endl;

    // *** You should not access db after this point ***

    // Client-side offline setup
    lwe_ann::LWESecretPack<int32_t> lwe_secret;
    lwe_ann::LWESecretPack<int32_t> lwe_secret_second;
    lwe_ann::LWEClient<int32_t> lwe_client(lwe_degree, 3.2);
    lwe_ann::LWEClient<int32_t> lwe_client_second(lwe_degree - 1, 3.2);
    lwe_ann::RLWEClient<uint64_t> rlwe_client(rlwe_degree, rlwe_modulus, 3.2, lwe_client.prng_factory_);

    int32_t ctxt_scale = 1 << 17;

    std::vector<std::vector<int32_t>> a_data_client(2);
    std::vector<std::vector<int32_t>> b_data_query(2);
#ifdef DEBUG_ZERO_A
    a_data_client[0].resize(padded_dim_first * lwe_degree);
    a_data_client[1].resize(padded_dim_mid * (lwe_degree - 1));
    std::fill(a_data_client[0].begin(), a_data_client[0].end(), 0);
    std::fill(a_data_client[1].begin(), a_data_client[1].end(), 0);
#else
    lwe_client.sample_a_from_seed(a_data_client[0], padded_dim_first * lwe_degree, pir.get_a_seed(0));
    lwe_client_second.sample_a_from_seed(a_data_client[1], padded_dim_mid * (lwe_degree - 1), pir.get_a_seed(1));
#endif
    lwe_ann::RLWESecret<uint64_t> rlwe_secret;
    rlwe_client.generate_secret(rlwe_secret);

    // Online stage
    // 1. client encryption (and query communication)

    // LWE secret sampling
    lwe_client.generate_secret_pack(lwe_secret, batch_size);
    lwe_secret_second.data_.resize((lwe_degree - 1) * batch_size);
    for (size_t b = 0; b < batch_size; b++)
    {
        std::copy(
            lwe_secret.data_.begin() + b * lwe_degree, lwe_secret.data_.begin() + (b + 1) * lwe_degree - 1,
            lwe_secret_second.data_.begin() + b * (lwe_degree - 1));
    }

    // Encryption
    lwe_client.compute_b_from_a(b_data_query[0], a_data_client[0], query_first, ctxt_scale, lwe_secret, batch_size);
    lwe_client_second.compute_b_from_a(
        b_data_query[1], a_data_client[1], query_second, 1 << 24, lwe_secret_second, batch_size);

    // Prepare RLWE conversion ciphertexts
    lwe_ann::RLWECiphertextPack<uint64_t> conversion_ctxts;
    size_t lwe_bits = sizeof(int32_t) * 8 - lwe_round_shift_bits;
    uint64_t rlwe_scale = rlwe_modulus / (1 << lwe_bits);
    rlwe_client.prepare_lwe_to_rlwe_conversion_ciphertexts(
        conversion_ctxts, lwe_secret, lwe_degree, rlwe_scale, rlwe_secret, false);

    // Just assume that the data is pinned
    std::cout << "Pinning query..." << std::endl;
    std::vector<int32_t *> query_pinned(2);
    cudastf::cuda_safe_call(cudaMallocHost(&query_pinned[0], b_data_query[0].size() * sizeof(int32_t)));
    memcpy(query_pinned[0], b_data_query[0].data(), b_data_query[0].size() * sizeof(int32_t));
    cudastf::cuda_safe_call(cudaMallocHost(&query_pinned[1], b_data_query[1].size() * sizeof(int32_t)));
    memcpy(query_pinned[1], b_data_query[1].data(), b_data_query[1].size() * sizeof(int32_t));
    std::cout << "Done!" << std::endl;

    // Conversion ciphertexts"
    std::cout << "Pinning conversion ciphertexts..." << std::endl;
    std::vector<uint64_t *> conversion_ctxts_pinned(2);
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
    // We ignore the time spent on the client-side
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
    size_t num_chunks = pir.get_num_rlwe_plain_word_chunks();
    uint64_t decrypt_scale = rlwe_scale * (1 << (24 - lwe_round_shift_bits));
    if (pir_res.size() != batch_size)
    {
        throw std::runtime_error("pir_res.size() != batch_size");
    }

    size_t pnnc = lwe_ann::pad_by<size_t>(padded_dim_last, rlwe_degree);
    for (size_t b = 0; b < batch_size; b++)
    {
        /* Test for debugging, checks whether the first RLWE-packed result is right.
        // Transpose the result
        if (pir_res[b].size() != 2 * num_chunks * pnnc * padded_dim_mid)
        {
            throw std::runtime_error("pir_res[b].size() != 2 * num_chunks * pnnc * padded_dim_mid");
        }
        std::vector<int8_t> tr_pir_res(pir_res[b].size());
        for (size_t ab = 0; ab < 2; ab++)
        {
            for (size_t i = 0; i < pnnc; i++)
            {
                for (size_t j = 0; j < padded_dim_mid; j++)
                {
                    for (size_t k = 0; k < num_chunks; k++)
                    {
                        int8_t val = pir_res[b]
                                            [k * padded_dim_mid + j + i * padded_dim_mid * num_chunks +
                                             ab * num_chunks * padded_dim_mid * pnnc];
                        tr_pir_res
                            [k + i * num_chunks + j * num_chunks * pnnc + ab * num_chunks * pnnc * padded_dim_mid] =
                                val;
                    }
                }
            }
        }
        // decrypt without padding
        auto tmp = lwe_ann::decrypt_pir_res_rlwe_2d<uint64_t, int8_t>(
            rlwe_client, rlwe_secret, tr_pir_res, rlwe_degree, 1, pnnc * padded_dim_mid, pnnc * padded_dim_mid,
            num_chunks, decrypt_scale, rlwe_round_shift_bits);
        if (tmp.size() != pnnc * padded_dim_mid)
        {
            throw std::runtime_error("tmp.size() != pnnc * padded_dim_mid");
        }

        // just extract a right cluster
        final_res.insert(
            final_res.end(), tmp.begin() + cluster_indices[b] * pnnc, tmp.begin() + cluster_indices[b] * pnnc +
        dim_last);
        */

        auto tmp = lwe_ann::decrypt_pir_res_rlwe_3d<uint64_t, int8_t>(
            rlwe_client, rlwe_secret, pir_res[b], rlwe_degree, dim_last, padded_dim_last, num_chunks, decrypt_scale,
            rlwe_round_shift_bits);
        final_res.insert(final_res.end(), tmp.begin(), tmp.end());
    }

    bool testPassed = true;
    int error_cnt = 0;
    for (size_t i = 0; i < dim_last * batch_size; i++)
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
