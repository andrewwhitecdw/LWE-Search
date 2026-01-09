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

#include "arith_utils.h"
#include "lwe_client.h"
#include "type_utils.h"

#include <seal/util/ntt.h>
#include <stdexcept>

namespace lwe_ann {
template <typename T> class PolynomialPack : public std::vector<T> {
  using Base = std::vector<T>;

public:
  PolynomialPack() = default;
  using Base::Base;

  // Disable copy but move-constructible
  PolynomialPack(PolynomialPack &&) = default;
  PolynomialPack &operator=(PolynomialPack &&) = default;
  ~PolynomialPack() = default;

  size_t batch_size_ = 1;
  bool is_ntt_ = false;
};

// A 'master' secret key is reused for RLWE encryption and decryption.
// It is in NTT-form by default.
template <typename uword> class RLWESecret {
  static_assert(is_valid_unsigned_word_v<uword>,
                "RLWESecret only supports unsigned words");

public:
  RLWESecret() = default;

  // Disable copy (and move)
  RLWESecret(const RLWESecret &) = delete;
  RLWESecret &operator=(const RLWESecret &) = delete;

  PolynomialPack<uword> data_;
};

template <typename uword> class RLWECiphertextPack {
  static_assert(is_valid_unsigned_word_v<uword>,
                "RLWECiphertext only supports unsigned words");

public:
  RLWECiphertextPack() = default;

  // Disable copy but move-constructible
  RLWECiphertextPack(RLWECiphertextPack &&) = default;
  ~RLWECiphertextPack() = default;

  PolynomialPack<uword> a_data_;
  PolynomialPack<uword> b_data_;

  double scale_ = 1.0;
  size_t round_shift_bits_ = 0;

  // prng_seed_type is std::array<std::uint64_t, prng_seed_uint64_count>;
  // a_seed_ is optional.
  seal::prng_seed_type a_seed_;
};

template <typename uword> class RLWEClient {
  static_assert(is_valid_unsigned_word_v<uword>,
                "RLWEClient only supports unsigned words");
  static_assert(std::is_same_v<uword, uint64_t>,
                "Currently, only uint64_t is supported");

  using Sec = RLWESecret<uword>;
  template <typename T> using Pt = PolynomialPack<T>;
  template <typename U> using Ct = RLWECiphertextPack<U>;

  using dword = make_dword_t<uword>;
  using sword = make_signed_t<uword>;

public:
  const size_t N_;
  const uword modulus_;
  const double sigma_;
  std::shared_ptr<seal::UniformRandomGeneratorFactory> prng_factory_;

private:
  barrett_util<uword> arith_;
  seal::util::NTTTables ntt_tables_;

  void check_input_params(size_t N, uword modulus) {
    if (!is_pow2(N)) {
      throw std::invalid_argument("RLWEClient: N must be a power of 2");
    }
    if ((modulus % (2 * N)) != 1) {
      throw std::invalid_argument("RLWEClient: modulus is not NTT-friendly");
    }
    if (modulus > std::numeric_limits<uword>::max() / 4) {
      throw std::invalid_argument("RLWEClient: modulus is too large");
    }
  }

  template <typename T>
  void scale_plaintext(std::vector<uword> &scaled_ptxt, const Pt<T> &ptxt,
                       double scale) const {
    static_assert(is_valid_signed_word_v<T> || std::is_floating_point_v<T>,
                  "Plaintext base type must be a valid signed word type or a "
                  "floating point type");
    if (scale < 1) {
      throw std::invalid_argument("scale must be greater than 1");
    }
    scaled_ptxt.resize(ptxt.size());

    sword w_scale = static_cast<sword>(std::llround(scale));
    std::transform(ptxt.begin(), ptxt.end(), scaled_ptxt.begin(), [&](auto x) {
      sword tmp;
      if constexpr (std::is_floating_point_v<T>) {
        tmp = static_cast<sword>(std::llround(x * scale));
      } else { // valid signed word type
        tmp = static_cast<sword>(x * w_scale);
      }
      return arith_.convert_to_unsigned_range(tmp);
    });
  }

  template <typename T>
  void descale_plaintext(std::vector<T> &ptxt,
                         const std::vector<uword> &scaled_ptxt, double scale,
                         size_t shift_bits) const {
    static_assert(is_valid_signed_word_v<T> || std::is_floating_point_v<T>,
                  "Plaintext base type must be a valid signed word type or a "
                  "floating point type");
    if (scale < 1) {
      throw std::invalid_argument("scale must be greater than 1");
    }
    uword shift_mult_factor = ((uword)1) << shift_bits;
    if (shift_mult_factor >= modulus_) {
      throw std::invalid_argument("rounding factor is too large");
    }
    ptxt.resize(scaled_ptxt.size());

    // Divide by scale and round to nearest
    sword w_scale = static_cast<sword>(std::llround(scale));
    uword half_scale = static_cast<uword>(w_scale >> 1);

    std::transform(scaled_ptxt.begin(), scaled_ptxt.end(), ptxt.begin(),
                   [&](uword x) -> T {
                     if (shift_bits > 0) {
                       x = arith_.mul(x, shift_mult_factor);
                     }
                     if constexpr (std::is_floating_point_v<T>) {
                       sword signed_pt_i = arith_.convert_to_signed_range(x);
                       return static_cast<T>(signed_pt_i / scale);
                     } else {
                       x += half_scale;
                       sword signed_pt_i = arith_.convert_to_signed_range(x);
                       return static_cast<T>(
                           signed_pt_i / w_scale +
                           (signed_pt_i >> (sizeof(sword) * 8 - 1)));
                     }
                   });
  }

  void add_random_noise(std::vector<uword> &data) const {
    auto prng = prng_factory_->create();

    // Adapted from SEAL.
    seal::RandomToStandardAdapter engine(prng);
    auto max_noise = sigma_ * 6;
    seal::util::ClippedNormalDistribution dist(0, sigma_, max_noise);

    sword modulus = static_cast<sword>(modulus_);
    std::for_each(data.begin(), data.end(), [&](auto &x) {
      sword sampled = static_cast<sword>(dist(engine));
      x = arith_.add(x, arith_.convert_to_unsigned_range(sampled));
    });
  }

  template <typename lwe_word>
  void prepare_lwe_to_rlwe_conversion_ciphertexts_compressed(
      RLWECiphertextPack<uword> &res,
      const LWESecretPack<lwe_word> &lwe_secret_pack, size_t lwe_n,
      double rlwe_scale, const RLWESecret<uword> &secret) {
    size_t batch_size = lwe_secret_pack.data_.size() / lwe_n;
    if (batch_size * lwe_n != lwe_secret_pack.data_.size()) {
      throw std::invalid_argument("RLWEClient: lwe_secret_pack.data_.size() "
                                  "must be a multiple of lwe_n");
    }

    size_t mult_factor = log2_ceil(lwe_n);
    sword w_scale = static_cast<sword>(std::llround(rlwe_scale));
    if (w_scale % mult_factor != 0) {
      throw std::invalid_argument(
          "RLWEClient: rlwe_scale must be a multiple of mult_factor");
    }
    w_scale /= mult_factor;

    PolynomialPack<lwe_word> lwe_secret_copied(N_ * batch_size, 0);
    for (size_t i = 0; i < batch_size; i++) {
      std::copy(lwe_secret_pack.data_.begin() + i * lwe_n,
                lwe_secret_pack.data_.begin() + (i + 1) * lwe_n,
                lwe_secret_copied.begin() + i * N_);
    }
    lwe_secret_copied.batch_size_ = batch_size;
    lwe_secret_copied.is_ntt_ = false;

    encrypt(res, lwe_secret_copied, w_scale, secret);
  }

public:
  void ntt(std::vector<uword> &data) const {
    size_t batch_size = data.size() / N_;
    if (data.size() != N_ * batch_size) {
      throw std::invalid_argument(
          "RLWEClient: data size is not N_ * batch_size_");
    }
    for (size_t i = 0; i < batch_size; i++) {
      seal::util::ntt_negacyclic_harvey(
          seal::util::CoeffIter(data.data() + i * N_), ntt_tables_);
    }
  }
  void inverse_ntt(std::vector<uword> &data) const {
    size_t batch_size = data.size() / N_;
    if (data.size() != N_ * batch_size) {
      throw std::invalid_argument(
          "RLWEClient: data size is not N_ * batch_size_");
    }
    for (size_t i = 0; i < batch_size; i++) {
      seal::util::inverse_ntt_negacyclic_harvey(
          seal::util::CoeffIter(data.data() + i * N_), ntt_tables_);
    }
  }

  RLWEClient(size_t N, uword modulus, double sigma)
      : N_(N), modulus_(modulus), sigma_(sigma), arith_(modulus),
        ntt_tables_{static_cast<int>(log2_ceil(N)), seal::Modulus(modulus)} {
    check_input_params(N, modulus);
    prng_factory_ = seal::UniformRandomGeneratorFactory::DefaultFactory();
  }

  RLWEClient(size_t N, uword modulus, double sigma,
             std::shared_ptr<seal::UniformRandomGeneratorFactory> prng_factory)
      : N_(N), modulus_(modulus), sigma_(sigma), arith_(modulus),
        ntt_tables_{static_cast<int>(log2_ceil(N)), seal::Modulus(modulus)},
        prng_factory_(prng_factory) {
    check_input_params(N, modulus);
  }

  // Adapted from LWEClient::generate_secret_pack
  void generate_secret(Sec &secret, bool is_ntt = true) const {
    // Use random (or default) seed for the secret PRNG.
    auto prng = prng_factory_->create();

    seal::seal_byte *entropy = new seal::seal_byte[N_];
    prng->generate(N_, entropy);
    secret.data_.resize(N_);

    constexpr seal::seal_byte mask_1{1};
    constexpr seal::seal_byte mask_2{2};

    uword minus_factor = modulus_ - 2;
    std::transform(entropy, entropy + N_, secret.data_.begin(),
                   [&](seal::seal_byte x) -> uword {
                     // This roughly produces 50% 0s, 25% 1s, and 25% -1s.
                     return (static_cast<uword>(x & mask_1) * minus_factor +
                             1) *
                            static_cast<uword>((x & mask_2) >> 1);
                   });
    delete[] entropy;

    // Perform NTT
    secret.data_.is_ntt_ = is_ntt;
    if (is_ntt) {
      ntt(secret.data_);
    }
  }

  void generate_seed(seal::prng_seed_type &seed) const {
    auto prng = prng_factory_->create();
    prng->generate(seal::prng_seed_byte_count,
                   reinterpret_cast<seal::seal_byte *>(seed.data()));
  }

  void sample_a_from_seed(Pt<uword> &a_data, size_t batch_size,
                          const seal::prng_seed_type &seed,
                          bool is_ntt = true) const {
    uword rejection_th =
        (std::numeric_limits<uword>::max() / modulus_) * modulus_;
    a_data.resize(batch_size * N_);
    a_data.batch_size_ = batch_size;

    auto prng =
        seal::UniformRandomGeneratorFactory::DefaultFactory()->create(seed);

    // Generate a_data with rejection sampling
    std::generate(a_data.begin(), a_data.end(), [&]() -> uword {
      uword x = rejection_th;
      while (x >= rejection_th) {
        prng->generate(sizeof(uword), reinterpret_cast<seal::seal_byte *>(&x));
      }
      return arith_.reduce_single(x);
    });

    // We do not need to explicitly perform NTT here.
    // We can interpret a as is_ntt or not as we like.
    a_data.is_ntt_ = is_ntt;
  }

  template <typename T>
  void compute_b_from_a(Pt<uword> &b_data, const Pt<uword> &a_data,
                        const Pt<T> &ptxt, double scale,
                        const Sec &secret) const {
    if (!a_data.is_ntt_) {
      throw std::invalid_argument("encrypt failed: a_data is not in NTT-form");
    }
    if (!secret.data_.is_ntt_) {
      throw std::invalid_argument("encrypt failed: secret is not in NTT-form");
    }
    if (ptxt.is_ntt_) {
      throw std::invalid_argument("encrypt failed: ptxt is in NTT-form");
    }

    size_t batch_size = a_data.batch_size_;
    size_t total_size = batch_size * N_;
    if (a_data.size() != ptxt.size() || a_data.size() != total_size ||
        ptxt.batch_size_ != batch_size || ptxt.size() != total_size) {
      throw std::invalid_argument(
          "RLWEClient: a_data and ptxt have different sizes");
    }

    // Prepare message and error parts
    scale_plaintext(b_data, ptxt, scale);
    add_random_noise(b_data);
    ntt(b_data);
    b_data.batch_size_ = batch_size;
    b_data.is_ntt_ = true;

    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < N_; j++) {
        uword tmp = arith_.mul(a_data[i * N_ + j], secret.data_[j]);
        b_data[i * N_ + j] = arith_.sub(b_data[i * N_ + j], tmp);
      }
    }
  }

  template <typename T>
  void encrypt(Ct<uword> &ctxt, const Pt<T> &ptxt, double scale,
               const Sec &secret) {
    generate_seed(ctxt.a_seed_);
    size_t batch_size = ptxt.size() / N_;
    if (batch_size * N_ != ptxt.size()) {
      throw std::invalid_argument("encrypt failed: invalid ptxt size");
    }
    sample_a_from_seed(ctxt.a_data_, batch_size, ctxt.a_seed_);
    compute_b_from_a(ctxt.b_data_, ctxt.a_data_, ptxt, scale, secret);
    ctxt.scale_ = scale;
    ctxt.round_shift_bits_ = 0;
  }

  template <typename T, typename U = uword>
  void decrypt(Pt<T> &ptxt, const Ct<U> &ctxt, const Sec &secret) const {
    static_assert(sizeof(U) <= sizeof(uword), "U must be smaller than uword");
    static_assert(std::is_floating_point_v<T> || is_valid_word_v<T>,
                  "Plaintext base type must be a valid word type or a floating "
                  "point type");

    std::vector<uword> tmp_a(ctxt.a_data_.begin(), ctxt.a_data_.end());

    size_t total_size = ctxt.a_data_.size();
    size_t batch_size = ctxt.a_data_.batch_size_;
    if (total_size != batch_size * N_ || total_size != ctxt.b_data_.size() ||
        batch_size != ctxt.b_data_.batch_size_) {
      throw std::invalid_argument("decrypt failed: invalid ctxt size");
    }
    ptxt.resize(total_size);

    // Convert to NTT form
    if (!ctxt.a_data_.is_ntt_) {
      ntt(tmp_a);
    }

    std::vector<uword> scaled_ptxt(total_size);

    if (!ctxt.b_data_.is_ntt_) {
      // If b_data is not in NTT form, we first compute a * secret only
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < N_; j++) {
          scaled_ptxt[i * N_ + j] =
              arith_.mul(tmp_a[i * N_ + j], secret.data_[j]);
        }
      }
    } else {
      // If b_data is in NTT form, we compute a + b * secret
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < N_; j++) {
          uword tmp = arith_.mul(tmp_a[i * N_ + j], secret.data_[j]);
          scaled_ptxt[i * N_ + j] = arith_.add(ctxt.b_data_[i * N_ + j], tmp);
        }
      }
    }

    // Perform INTT
    inverse_ntt(scaled_ptxt);

    // Add b_data here.
    if (!ctxt.b_data_.is_ntt_) {
      std::transform(scaled_ptxt.begin(), scaled_ptxt.end(),
                     ctxt.b_data_.begin(), scaled_ptxt.begin(),
                     [&](uword x, uword b) { return arith_.add(x, b); });
    }

    // Divide by scale and round to nearest
    descale_plaintext(ptxt, scaled_ptxt, ctxt.scale_, ctxt.round_shift_bits_);
    ptxt.batch_size_ = batch_size;
    ptxt.is_ntt_ = false;
  }

  template <typename lwe_word>
  void prepare_lwe_to_rlwe_conversion_ciphertexts(
      RLWECiphertextPack<uword> &res,
      const LWESecretPack<lwe_word> &lwe_secret_pack, size_t lwe_n,
      double rlwe_scale, const RLWESecret<uword> &secret,
      bool compressed = false) {
    // Do some basic checks.
    if (lwe_n > N_) {
      throw std::invalid_argument(
          "RLWEClient: lwe_n must be less than or equal to N_");
    }
    size_t batch_size = lwe_secret_pack.data_.size() / lwe_n;
    if (batch_size * lwe_n != lwe_secret_pack.data_.size()) {
      throw std::invalid_argument("RLWEClient: lwe_secret_pack.data_.size() "
                                  "must be a multiple of lwe_n");
    }

    if (compressed) {
      prepare_lwe_to_rlwe_conversion_ciphertexts_compressed(
          res, lwe_secret_pack, lwe_n, rlwe_scale, secret);
      return;
    }

    PolynomialPack<lwe_word> lwe_secret_copied(N_ * batch_size * lwe_n, 0);
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < lwe_n; j++) {
        // The data organization may be modified later.
        // The current ordering is N_ -> lwe_n -> batch_size.
        lwe_secret_copied[N_ * (i * lwe_n + j)] =
            lwe_secret_pack.data_[i * lwe_n + j];
      }
    }
    lwe_secret_copied.batch_size_ = batch_size * lwe_n;
    lwe_secret_copied.is_ntt_ = false;

    encrypt(res, lwe_secret_copied, rlwe_scale, secret);
  }

  // Disable copy (and move)
  RLWEClient(const RLWEClient &) = delete;
  RLWEClient &operator=(const RLWEClient &) = delete;
};

} // namespace lwe_ann
