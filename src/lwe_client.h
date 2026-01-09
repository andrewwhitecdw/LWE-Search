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

#include "seal/randomgen.h"
#include "seal/randomtostd.h"
#include "seal/util/clipnormal.h"
#include "type_utils.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lwe_ann {
template <typename word> class LWESecretPack {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");

public:
  // matrix stored in column-major order
  std::vector<word> data_; // n_ x batch_size matrix
  LWESecretPack() = default;

  // Disable copy (and move)
  LWESecretPack(const LWESecretPack &) = delete;
  LWESecretPack &operator=(const LWESecretPack &) = delete;
};

template <typename word> class LWECiphertextPack {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");

public:
  // matrices stored in column-major order
  // No shape information is stored
  std::vector<word> a_data_; // ptxt_len x n_ matrix
  std::vector<word> b_data_; // ptxt_len x batch_size matrix

  double scale_;
  size_t round_shift_bits_ = 0;

  // prng_seed_type is std::array<std::uint64_t, prng_seed_uint64_count>;
  seal::prng_seed_type a_seed_;

  LWECiphertextPack() = default;

  // Disable copy but move-constructible
  LWECiphertextPack(LWECiphertextPack &&) = default;
  ~LWECiphertextPack() = default;
};

template <typename word> class LWEClient {
  static_assert(is_valid_signed_word_v<word>,
                "word must be a valid signed type");

protected:
  void add_random_noise(std::vector<word> &data) const {
    auto prng = prng_factory_->create();

    // Adapted from SEAL.
    seal::RandomToStandardAdapter engine(prng);
    auto max_noise = sigma_ * 6;
    seal::util::ClippedNormalDistribution dist(0, sigma_, max_noise);

    std::for_each(data.begin(), data.end(), [&dist, &engine](auto &x) {
      x += static_cast<word>(dist(engine));
    });
  }

  using Ct = LWECiphertextPack<word>;
  template <typename T> using Pt = std::vector<T>;
  using Sec = LWESecretPack<word>;

  template <typename T>
  void scale_plaintext(std::vector<word> &scaled_pt, const Pt<T> &ptxt,
                       double scale) const {
    static_assert(is_valid_signed_word_v<T> || std::is_floating_point_v<T>,
                  "Plaintext base type must be a valid signed word type or a "
                  "floating point type");
    if (scale < 1) {
      throw std::invalid_argument("scale must be greater than 1");
    }
    scaled_pt.resize(ptxt.size());

    if constexpr (std::is_floating_point_v<T>) {
      std::transform(ptxt.begin(), ptxt.end(), scaled_pt.begin(), [&](auto x) {
        return static_cast<word>(std::llround(x * scale));
      });
    } else { // valid signed word type
      word w_scale = static_cast<word>(std::llround(scale));
      std::transform(ptxt.begin(), ptxt.end(), scaled_pt.begin(),
                     [&](auto x) { return static_cast<word>(x * w_scale); });
    }
  }

  template <typename T>
  void descale_plaintext(std::vector<T> &res,
                         const std::vector<word> &scaled_pt, double scale,
                         size_t shift_bits) const {
    static_assert(is_valid_signed_word_v<T> || std::is_floating_point_v<T>,
                  "Plaintext base type must be a valid signed word type or a "
                  "floating point type");
    if (scale < 1) {
      throw std::invalid_argument("scale must be greater than 1");
    }
    if (shift_bits >= sizeof(word) * 8) {
      throw std::invalid_argument("rounding factor is too large");
    }
    res.resize(scaled_pt.size());

    word w_scale = static_cast<word>(std::llround(scale));
    word half_scale = w_scale >> 1;

    std::transform(
        scaled_pt.begin(), scaled_pt.end(), res.begin(), [&](word x) -> T {
          x <<= shift_bits;
          if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(x / scale);
          } else { // valid signed word type
            x += half_scale;
            return static_cast<T>(x / w_scale + (x >> (sizeof(word) * 8 - 1)));
          }
        });
  }

public:
  const size_t n_;
  const double sigma_;
  std::shared_ptr<seal::UniformRandomGeneratorFactory> prng_factory_;

  LWEClient(size_t n, double sigma) : n_(n), sigma_(sigma) {
    // PRNG factory will be used to sample secrets, errors, and a-seeds
    // But not directly used for the actual a_data_
    prng_factory_ = seal::UniformRandomGeneratorFactory::DefaultFactory();
  }

  LWEClient(size_t n, double sigma,
            std::shared_ptr<seal::UniformRandomGeneratorFactory> prng_factory)
      : n_(n), sigma_(sigma), prng_factory_(prng_factory) {}

  void generate_secret_pack(Sec &secret, size_t batch_size = 1) const {
    // Use random (or default) seed for the secret PRNG.
    auto prng = prng_factory_->create();

    size_t num_words = batch_size * n_;

    seal::seal_byte *entropy = new seal::seal_byte[num_words];
    prng->generate(num_words, entropy);
    secret.data_.resize(num_words);

    constexpr seal::seal_byte mask_1{1};
    constexpr seal::seal_byte mask_2{2};

    std::transform(entropy, entropy + num_words, secret.data_.begin(),
                   [&](seal::seal_byte x) -> word {
                     // This roughly produces 50% 0s, 25% 1s, and 25% -1s.
                     return static_cast<word>(x & mask_1) -
                            static_cast<word>((x & mask_2) >> 1);
                   });
    delete[] entropy;
  }

  void generate_seed(seal::prng_seed_type &seed) const {
    auto prng = prng_factory_->create();
    prng->generate(seal::prng_seed_byte_count,
                   reinterpret_cast<seal::seal_byte *>(seed.data()));
  }

  void sample_a_from_seed(std::vector<word> &a_data, size_t a_size,
                          const seal::prng_seed_type &seed) const {
    a_data.resize(a_size);
    auto prng =
        seal::UniformRandomGeneratorFactory::DefaultFactory()->create(seed);
    prng->generate(a_size * sizeof(word),
                   reinterpret_cast<seal::seal_byte *>(a_data.data()));
  }

  template <typename T>
  void compute_b_from_a(std::vector<word> &b_data,
                        const std::vector<word> &a_data, const Pt<T> &ptxt,
                        double scale, const Sec &secret,
                        size_t batch_size = 1) const {
    // Check size of input
    // Check size of input
    size_t ptxt_len = ptxt.size() / batch_size;
    size_t a_size = ptxt_len * n_;
    size_t secret_size = n_ * batch_size;
    size_t b_size = ptxt_len * batch_size;
    if (ptxt_len * batch_size != ptxt.size() ||
        secret.data_.size() < secret_size || a_data.size() != a_size) {
      throw std::invalid_argument("encryption failed: size mismatch");
    }

    b_data.resize(b_size);
    std::vector<word> scaled_pt(b_size);
    scale_plaintext(scaled_pt, ptxt, scale);
    add_random_noise(scaled_pt);

    // Just an unoptimized CPU GEMM implementation
    // Implemented in a naive way as the data layout can be modified
    // later. Currently, assuming column-major order for everything
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < ptxt_len; j++) {
        word &b_ij = b_data[i * ptxt_len + j];
        b_ij = scaled_pt[i * ptxt_len + j];
        for (size_t k = 0; k < n_; k++) {
          b_ij -= secret.data_[i * n_ + k] * a_data[k * ptxt_len + j];
        }
      }
    }
  }

  /**
   * @brief Encrypt a plaintext matrix into a ciphertext vector
   *
   * @param ctxt encryption result
   * @param ptxt matrix to encrypt
   * @param scale scale to be multiplied to the plaintext
   * @param secret secret key pack
   * @param batch_size batch size of ptxt and secret
   */
  template <typename T>
  void encrypt(Ct &ctxt, const Pt<T> &ptxt, double scale, const Sec &secret,
               size_t batch_size = 1) const {
    generate_seed(ctxt.a_seed_);
    size_t ptxt_len = ptxt.size() / batch_size;
    sample_a_from_seed(ctxt.a_data_, ptxt_len * n_, ctxt.a_seed_);
    compute_b_from_a(ctxt.b_data_, ctxt.a_data_, ptxt, scale, secret,
                     batch_size);
    ctxt.scale_ = scale;
  }

  template <typename T, typename U = word>
  void decrypt(Pt<T> &ptxt, const LWECiphertextPack<U> &ctxt,
               const Sec &secret) const {
    static_assert(sizeof(U) <= sizeof(word),
                  "Decrypt: LWE ciphertext cannot use larger word type");

    auto &a_data = ctxt.a_data_;
    auto &b_data = ctxt.b_data_;
    auto &s_data = secret.data_;

    size_t a_size = a_data.size();
    size_t b_size = b_data.size();

    size_t ptxt_len = a_size / n_;
    size_t batch_size = b_size / ptxt_len;
    if (a_size != n_ * ptxt_len || b_size != ptxt_len * batch_size ||
        s_data.size() != n_ * batch_size) {
      throw std::invalid_argument("decryption failed: size mismatch");
    }

    ptxt.resize(b_size);
    std::vector<word> scaled_pt(b_size);

    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < ptxt_len; j++) {
        word &pt_ij = scaled_pt[i * ptxt_len + j];
        pt_ij = static_cast<word>(b_data[i * ptxt_len + j]);
        for (size_t k = 0; k < n_; k++) {
          pt_ij += s_data[i * n_ + k] * a_data[k * ptxt_len + j];
        }
      }
    }

    descale_plaintext(ptxt, scaled_pt, ctxt.scale_, ctxt.round_shift_bits_);
  }
};

} // namespace lwe_ann
