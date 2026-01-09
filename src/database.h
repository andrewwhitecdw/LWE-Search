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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace lwe_ann {

template <typename index_t>
inline double calculate_recall(std::vector<index_t> &gt,
                               std::vector<index_t> &res, index_t num_vectors) {
  index_t recall_k = res.size() / num_vectors;
  index_t gt_k = gt.size() / num_vectors;
  double total_recall = 0.0;

  for (int query_id = 0; query_id < num_vectors; query_id++) {
    // Build hash set of ground truth neighbors for O(1) lookup
    std::unordered_set<index_t> gt_set;
    for (uint32_t gt_idx = 0; gt_idx < recall_k; gt_idx++) {
      gt_set.insert(gt[query_id * gt_k + gt_idx]);
    }

    // Count matches between predicted and ground truth
    int matches = 0;
    for (uint32_t pred_idx = 0; pred_idx < recall_k; pred_idx++) {
      if (gt_set.find(res[query_id * recall_k + pred_idx]) != gt_set.end()) {
        matches++;
      }
    }

    double query_recall = static_cast<double>(matches) / recall_k;
    total_recall += query_recall;
  }

  return total_recall / num_vectors;
}

template <typename T>
inline void read_bytes(std::ifstream &file, T *dst, size_t size) {
  static constexpr size_t read_granularity_ = 1024 * 1024; // 1MB
  char *dst_ptr = reinterpret_cast<char *>(dst);
  while (size > 0) {
    size_t to_read = std::min(size, read_granularity_);
    file.read(dst_ptr, to_read);
    dst_ptr += to_read;
    size -= to_read;
  }
}

inline void print_remaining_bytes(std::ifstream &file) {
  size_t bytes = 0;
  char tmp;
  while (!file.eof()) {
    file.read(&tmp, 1);
    bytes++;
  }
  std::cout << "Remaining bytes: " << bytes << " bytes" << std::endl;
}

inline void check_end_of_file(std::ifstream &file,
                              const std::string &filename) {
  // Checks if the next byte to read is eof or the second next byte is eof
  if (file.eof()) {
    throw std::runtime_error("eof encountered before the end of the file");
  }

  if (file.peek() == std::ifstream::traits_type::eof()) {
    return;
  }
  std::cout << "WARNING: " << filename << " is not sized correctly"
            << std::endl;
  print_remaining_bytes(file);
}

template <typename index_t, typename plain_word> struct Database {
  struct ClusterData {
    index_t num_records_ = 0;
    plain_word *centroid_ = nullptr;
    index_t *record_indices_ = nullptr;
    plain_word *records_ = nullptr;
  };

  struct MesoclusterData {
    index_t num_clusters_ = 0;
    plain_word *centroid_ = nullptr;
    ClusterData *cluster_data_ = nullptr;
    index_t effective_num_clusters_ = 0;
  };

  index_t num_mesoclusters_ = 0;
  index_t num_clusters_ = 0;
  index_t num_records_ = 0;
  index_t vector_len_ = 0;

  std::vector<MesoclusterData> mesocluster_data_;
  std::vector<plain_word> mesocluster_centroids_buffer_;

  std::vector<ClusterData> cluster_data_buffer_;
  std::vector<plain_word> cluster_centroids_buffer_;
  std::vector<index_t> record_indices_buffer_;
  std::vector<plain_word> records_buffer_;

  // For 2D: not used
  // For 3D: maps plain_mesocluster_idx -> centroid
  std::vector<plain_word> processed_mesocluster_centroids_;

  // For 2D: maps plain_cluster_idx -> centroid
  // For 3D: maps (eff_dim_first_idx, plain_cluster_idx % dim_mid) -> centroid
  std::vector<plain_word> processed_centroids_;

  // For 2D: maps (eff_dim_first_idx, plain_record_idx % dim_mid) -> record
  // For 3D: maps (eff_dim_first_idx, eff_dim_mid_idx, plain_record_idx %
  // dim_last) -> record
  std::vector<plain_word> processed_records_;
  std::vector<index_t> processed_dims_;

  // cluster_dim_splits_[i] means that
  std::vector<index_t> extra_mesocluster_dim_splits_;
  std::vector<std::vector<index_t>> extra_cluster_dim_splits_;

  template <typename T> void update_scalar(T &dst, T &src) {
    if (dst != 0) {
      if (dst != src) {
        throw std::runtime_error("dst (" + std::to_string(dst) + ") != src (" +
                                 std::to_string(src) + ")");
      }
    } else {
      dst = src;
    }
  }

  template <typename V>
  inline std::vector<std::pair<index_t, index_t>>
  find_best_score_indices(const std::vector<V> &scores, index_t k,
                          index_t dim_last) {
    auto cmp = [](const std::tuple<V, index_t, index_t> &a,
                  const std::tuple<V, index_t, index_t> &b) {
      return std::get<0>(a) > std::get<0>(b);
    };

    if (scores.size() < k + 1) {
      throw std::runtime_error("scores.size() < k");
    }

    std::vector<std::tuple<V, index_t, index_t>> high_scores(k + 1);

    // Initialize
    for (index_t i = 0; i < k + 1; i++) {
      high_scores[i] = std::make_tuple(scores[i], i / dim_last, i % dim_last);
    }
    std::make_heap(high_scores.begin(), high_scores.end(), cmp);
    std::pop_heap(high_scores.begin(), high_scores.end(), cmp);

    index_t num_records = scores.size();
    if (num_records % dim_last != 0) {
      throw std::runtime_error("num_records % dim_last != 0");
    }

    for (index_t i = k + 1; i < num_records; i++) {
      if (scores[i] > std::get<0>(high_scores.front())) {
        high_scores.back() =
            std::make_tuple(scores[i], i / dim_last, i % dim_last);
        std::pop_heap(high_scores.begin(), high_scores.end(), cmp);
      }
    }
    std::sort(high_scores.begin(), high_scores.end(), cmp);
    std::vector<std::pair<index_t, index_t>> indices(k);
    for (size_t i = 0; i < k; i++) {
      indices[i] = std::make_pair(std::get<1>(high_scores[i]),
                                  std::get<2>(high_scores[i]));
    }
    return indices;
  }

  template <typename T, typename U, typename V = T>
  inline std::vector<index_t>
  find_nearest_neighbors(const std::vector<T> &query,
                         const std::vector<U> &records, index_t k) {
    auto cmp = [](const std::pair<V, index_t> &a,
                  const std::pair<V, index_t> &b) { return a.first > b.first; };
    if (query.size() != vector_len_) {
      throw std::runtime_error("query.size() != vector_len_");
    }
    if (records.size() % vector_len_ != 0) {
      throw std::runtime_error("records.size() % vector_len_ != 0");
    }
    index_t num_records = records.size() / vector_len_;
    if (num_records < k + 1) {
      throw std::runtime_error("num_records < k + 1");
    }

    // Use min heap to find the k records with the highest inner product values
    std::vector<std::pair<V, index_t>> scores(k + 1);
    for (index_t i = 0; i < k + 1; i++) {
      V s = 0;
      for (index_t j = 0; j < vector_len_; j++) {
        s += static_cast<V>(query[j]) * records[i * vector_len_ + j];
      }
      scores[i] = std::make_pair(s, i);
    }
    std::make_heap(scores.begin(), scores.end(), cmp);
    std::pop_heap(scores.begin(), scores.end(), cmp);
    for (index_t i = k + 1; i < num_records; i++) {
      V s = 0;
      for (index_t j = 0; j < vector_len_; j++) {
        s += static_cast<V>(query[j]) * records[i * vector_len_ + j];
      }
      if (s > scores.front().first) {
        scores.back() = std::make_pair(s, i);
        std::pop_heap(scores.begin(), scores.end(), cmp);
      }
    }

    // Order: descending
    std::sort(scores.begin(), scores.end(), cmp);
    std::vector<index_t> indices(k);
    for (size_t i = 0; i < k; i++) {
      indices[i] = scores[i].second;
    }
    return indices;
  }

  void parse_num_clusters_per_mesocluster(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    index_t num_meso;
    read_bytes(file, &num_meso, sizeof(index_t));

    std::cout << "Number of mesoclusters: " << num_meso << std::endl;

    update_scalar(num_mesoclusters_, num_meso);

    mesocluster_data_.resize(num_meso);

    std::vector<index_t> num_clusters_per_mesocluster(num_meso);
    read_bytes(file, num_clusters_per_mesocluster.data(),
               sizeof(index_t) * num_meso);
    index_t total_num_clusters = std::accumulate(
        num_clusters_per_mesocluster.begin(),
        num_clusters_per_mesocluster.end(), 0, std::plus<index_t>());

    std::cout << "Number of clusters: " << total_num_clusters << std::endl;

    update_scalar(num_clusters_, total_num_clusters);

    cluster_data_buffer_.clear();
    cluster_data_buffer_.resize(total_num_clusters);

    index_t nc_idx = 0;
    for (index_t i = 0; i < num_meso; i++) {
      index_t nc = num_clusters_per_mesocluster.at(i);
      mesocluster_data_.at(i).num_clusters_ = nc;
      mesocluster_data_.at(i).centroid_ = nullptr;
      mesocluster_data_.at(i).cluster_data_ =
          cluster_data_buffer_.data() + nc_idx;
      nc_idx += nc;
    }

    check_end_of_file(file, filename);
    file.close();
  }

  void parse_mesocluster_centroids(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    index_t num_meso, veclen;
    read_bytes(file, &num_meso, sizeof(index_t));
    read_bytes(file, &veclen, sizeof(index_t));

    std::cout << "Number of mesoclusters: " << num_meso << std::endl;
    std::cout << "Vector length: " << veclen << std::endl;

    update_scalar(num_mesoclusters_, num_meso);
    update_scalar(vector_len_, veclen);

    mesocluster_centroids_buffer_.resize(num_meso * veclen);
    read_bytes(file, mesocluster_centroids_buffer_.data(),
               sizeof(plain_word) * num_meso * veclen);

    for (index_t i = 0; i < num_meso; i++) {
      auto &mesocluster_data = mesocluster_data_.at(i);
      mesocluster_data.centroid_ =
          mesocluster_centroids_buffer_.data() + i * veclen;
    }

    check_end_of_file(file, filename);
    file.close();
  }

  void parse_index(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    index_t num_cl, veclen;
    read_bytes(file, &num_cl, sizeof(index_t));
    read_bytes(file, &veclen, sizeof(index_t));

    std::cout << "Number of clusters: " << num_cl << std::endl;
    std::cout << "Vector length: " << veclen << std::endl;

    update_scalar(num_clusters_, num_cl);
    update_scalar(vector_len_, veclen);

    // cluster_data_buffer_.resize(num_cl);
    if (cluster_data_buffer_.size() != num_cl) {
      throw std::runtime_error("cluster_data_buffer_.size() != num_cl");
    }
    cluster_centroids_buffer_.clear();
    record_indices_buffer_.clear();
    records_buffer_.clear();

    this->num_records_ = 0;
    for (index_t i = 0; i < num_cl; i++) {
      auto &cl = cluster_data_buffer_.at(i);
      index_t num_records;
      read_bytes(file, &num_records, sizeof(index_t));
      update_scalar(cl.num_records_, num_records);
      this->num_records_ += num_records;

      std::vector<plain_word> centroid(veclen);
      std::vector<index_t> record_indices(num_records);
      std::vector<plain_word> records(num_records * veclen);

      read_bytes(file, centroid.data(), sizeof(plain_word) * veclen);
      read_bytes(file, record_indices.data(), sizeof(index_t) * num_records);
      read_bytes(file, records.data(),
                 sizeof(plain_word) * veclen * num_records);

      cluster_centroids_buffer_.insert(cluster_centroids_buffer_.end(),
                                       centroid.begin(), centroid.end());
      record_indices_buffer_.insert(record_indices_buffer_.end(),
                                    record_indices.begin(),
                                    record_indices.end());
      records_buffer_.insert(records_buffer_.end(), records.begin(),
                             records.end());
    }
    std::cout << "Total number of records: " << this->num_records_ << std::endl;

    // Register pointers to cluster_centroids_buffer_
    index_t record_idx = 0;
    for (index_t i = 0; i < num_cl; i++) {
      auto &cl = cluster_data_buffer_.at(i);
      cl.centroid_ = cluster_centroids_buffer_.data() + i * veclen;
      cl.record_indices_ = record_indices_buffer_.data() + record_idx;
      cl.records_ = records_buffer_.data() + record_idx * veclen;
      record_idx += cl.num_records_;
    }

    check_end_of_file(file, filename);
    file.close();
  }

  inline std::pair<index_t, index_t>
  to_eff_idx_worker(const std::vector<index_t> &splits,
                    index_t plain_idx) const {
    index_t eff_idx = 0;
    index_t prev_cumul_clusters = 0;
    index_t current_split_minus_one = splits.size() - 1;
    for (; current_split_minus_one > 0; current_split_minus_one--) {
      index_t num_cumul_clusters = splits.at(current_split_minus_one);
      index_t num_clusters_with_this_split =
          num_cumul_clusters - prev_cumul_clusters; // 1
      prev_cumul_clusters = num_cumul_clusters;     // 1
      if (plain_idx >= num_clusters_with_this_split) {
        eff_idx += (current_split_minus_one + 1) * num_clusters_with_this_split;
        plain_idx -= num_clusters_with_this_split;
        // continue to update current_split_minus_one
      } else {
        eff_idx += (current_split_minus_one + 1) * plain_idx;
        plain_idx = 0;
        break; // break without updating current_split_minus_one
      }
    }

    if (plain_idx > 0) {
      // for num_splits = 1 leftovers
      eff_idx += plain_idx;
    }
    return std::make_pair(eff_idx, current_split_minus_one + 1);
  }

  inline std::pair<index_t, index_t>
  to_eff_cluster_idx(index_t plain_cluster_idx, index_t plain_meso_idx = 0) {
    return to_eff_idx_worker(extra_cluster_dim_splits_.at(plain_meso_idx),
                             plain_cluster_idx);
  }

  std::pair<index_t, index_t> to_eff_mesocluster_idx(index_t plain_meso_idx) {
    return to_eff_idx_worker(extra_mesocluster_dim_splits_, plain_meso_idx);
  }

  void prepare_ann_2d(index_t dim_last) {
    // Sort in descending order of num_records_
    std::sort(cluster_data_buffer_.begin(), cluster_data_buffer_.end(),
              [](const ClusterData &a, const ClusterData &b) {
                return a.num_records_ > b.num_records_;
              });

    index_t max_dim = cluster_data_buffer_.front().num_records_;
    if (max_dim < dim_last) {
      std::cout << "WARNING: using too large dim_last" << std::endl;
    }

    index_t max_dim_split = (max_dim + dim_last - 1) / dim_last;
    extra_cluster_dim_splits_.clear();
    extra_cluster_dim_splits_.resize(1);
    extra_cluster_dim_splits_.at(0).resize(max_dim_split, 0);

    index_t eff_dim_first = 0;
    for (auto &cl : cluster_data_buffer_) {
      index_t num_splits = (cl.num_records_ + dim_last - 1) / dim_last;
      extra_cluster_dim_splits_.at(0).at(num_splits - 1)++;
      eff_dim_first += num_splits;
    }

    // Make extra_cluster_dim_splits_ cumulative
    for (index_t i = extra_cluster_dim_splits_.at(0).size() - 1; i > 0; i--) {
      extra_cluster_dim_splits_.at(0).at(i - 1) +=
          extra_cluster_dim_splits_.at(0).at(i);
    }

    processed_records_.clear();
    processed_records_.resize(eff_dim_first * dim_last * vector_len_);
    processed_centroids_.clear();
    processed_centroids_.resize(num_clusters_ * vector_len_);

    for (index_t i = 0; i < num_clusters_; i++) {
      auto [eff_dim_first_idx, _] = to_eff_cluster_idx(i);
      auto &cl = cluster_data_buffer_.at(i);

      // Copy centroids to processed_centroids_
      std::copy(cl.centroid_, cl.centroid_ + vector_len_,
                processed_centroids_.begin() + i * vector_len_);

      // Copy records to processed_records_
      index_t eff_dim_first_inner_idx = 0;
      index_t eff_dim_last_idx = 0;
      for (index_t j = 0; j < cl.num_records_; j++) {
        std::copy(cl.records_ + j * vector_len_,
                  cl.records_ + (j + 1) * vector_len_,
                  processed_records_.begin() +
                      (eff_dim_first_idx + eff_dim_first_inner_idx +
                       eff_dim_last_idx * eff_dim_first) *
                          vector_len_);
        // Update eff_dim_first_inner_idx and eff_dim_last_idx
        eff_dim_last_idx++;
        if (eff_dim_last_idx == dim_last) {
          eff_dim_first_inner_idx++;
          eff_dim_last_idx = 0;
        }
      }
    }

    processed_dims_ =
        std::vector<index_t>{eff_dim_first * vector_len_, dim_last};

    // Calculate storage overhead
    double overhead =
        (static_cast<double>(processed_records_.size()) / num_records_) /
        vector_len_;
    std::cout << "Storage overhead for the main database: " << overhead
              << std::endl;
  }

  void prepare_ann_3d(index_t dim_mid, index_t dim_last) {
    extra_cluster_dim_splits_.resize(num_mesoclusters_);
    index_t global_max_dim_last = 0;
    index_t nc_idx = 0;
    index_t total_num_records = 0;
    for (index_t i = 0; i < num_mesoclusters_; i++) {
      auto &ms = mesocluster_data_.at(i);
      index_t nc = ms.num_clusters_;
      auto cluster_data_begin = cluster_data_buffer_.begin() + nc_idx;
      auto cluster_data_end = cluster_data_begin + nc;

      std::sort(cluster_data_begin, cluster_data_end,
                [](const ClusterData &a, const ClusterData &b) {
                  return a.num_records_ > b.num_records_;
                });

      index_t max_dim_last = cluster_data_begin->num_records_;
      global_max_dim_last = std::max(global_max_dim_last, max_dim_last);
      index_t max_dim_last_split = (max_dim_last + dim_last - 1) / dim_last;

      // Populating extra_cluster_dim_splits_
      extra_cluster_dim_splits_.at(i).resize(max_dim_last_split, 0);
      ms.effective_num_clusters_ = 0;
      for (auto cl_it = cluster_data_begin; cl_it != cluster_data_end;
           cl_it++) {
        index_t num_splits = (cl_it->num_records_ + dim_last - 1) / dim_last;
        extra_cluster_dim_splits_.at(i).at(num_splits - 1)++;
        ms.effective_num_clusters_ += num_splits;
        total_num_records += cl_it->num_records_;
      }
      // Reverse cumulative representation
      for (index_t j = extra_cluster_dim_splits_.at(i).size() - 1; j > 0; j--) {
        extra_cluster_dim_splits_.at(i).at(j - 1) +=
            extra_cluster_dim_splits_.at(i).at(j);
      }
      nc_idx += nc;
    }

    if (global_max_dim_last < dim_last) {
      std::cout << "WARNING: using too large dim_last" << std::endl;
    }

    // Sort in the order of descending effective_num_clusters_
    std::sort(mesocluster_data_.begin(), mesocluster_data_.end(),
              [](const MesoclusterData &a, const MesoclusterData &b) {
                return a.effective_num_clusters_ > b.effective_num_clusters_;
              });

    index_t max_dim_mid = mesocluster_data_.front().effective_num_clusters_;
    if (max_dim_mid < dim_mid) {
      std::cout << "WARNING: using too large dim_mid" << std::endl;
    }
    index_t max_dim_mid_split = (max_dim_mid + dim_mid - 1) / dim_mid;

    extra_mesocluster_dim_splits_.clear();
    extra_mesocluster_dim_splits_.resize(max_dim_mid_split);

    index_t eff_dim_first = 0;
    for (index_t i = 0; i < num_mesoclusters_; i++) {
      auto &ms = mesocluster_data_.at(i);
      index_t num_splits = (ms.effective_num_clusters_ + dim_mid - 1) / dim_mid;
      extra_mesocluster_dim_splits_.at(num_splits - 1)++;
      eff_dim_first += num_splits;
    }

    // Reverse cumulative representation
    for (index_t i = extra_mesocluster_dim_splits_.size() - 1; i > 0; i--) {
      extra_mesocluster_dim_splits_.at(i - 1) +=
          extra_mesocluster_dim_splits_.at(i);
    }

    processed_mesocluster_centroids_.clear();
    processed_mesocluster_centroids_.resize(num_mesoclusters_ * vector_len_);
    processed_centroids_.clear();
    processed_centroids_.resize(eff_dim_first * dim_mid * vector_len_);
    processed_records_.clear();
    processed_records_.resize(eff_dim_first * dim_mid * dim_last * vector_len_);

    for (index_t i = 0; i < num_mesoclusters_; i++) {
      auto &ms = mesocluster_data_.at(i);
      auto [eff_ms_idx, _] = to_eff_mesocluster_idx(i);
      // Copy mesocluster centroids to processed_mesocluster_centroids_
      std::copy(ms.centroid_, ms.centroid_ + vector_len_,
                processed_mesocluster_centroids_.begin() + i * vector_len_);

      // Copy cluster centroids to processed_centroids_
      for (index_t j = 0; j < ms.num_clusters_; j++) {
        auto &cl = ms.cluster_data_[j];
        index_t eff_dim_first_idx = eff_ms_idx + j / dim_mid;
        index_t eff_dim_mid_idx = j % dim_mid;
        std::copy(cl.centroid_, cl.centroid_ + vector_len_,
                  processed_centroids_.begin() +
                      (eff_dim_first_idx + eff_dim_mid_idx * eff_dim_first) *
                          vector_len_);
      }

      // Copy records to processed_records_
      for (index_t j = 0; j < ms.num_clusters_; j++) {
        auto &cl = ms.cluster_data_[j];
        auto [eff_cl_idx, _] = to_eff_cluster_idx(j, i);
        for (index_t k = 0; k < cl.num_records_; k++) {
          index_t eff_dim_last_idx = k % dim_last;
          index_t dm_tmp = (k / dim_last) + eff_cl_idx;
          index_t eff_dim_mid_idx = dm_tmp % dim_mid;
          index_t eff_dim_first_idx = eff_ms_idx + dm_tmp / dim_mid;
          std::copy(cl.records_ + k * vector_len_,
                    cl.records_ + (k + 1) * vector_len_,
                    processed_records_.begin() +
                        (eff_dim_first_idx + eff_dim_mid_idx * eff_dim_first +
                         eff_dim_last_idx * dim_mid * eff_dim_first) *
                            vector_len_);
        }
      }
    }

    processed_dims_ =
        std::vector<index_t>{eff_dim_first * vector_len_, dim_mid, dim_last};

    double overhead =
        (static_cast<double>(processed_records_.size()) / total_num_records) /
        vector_len_;
    std::cout << "Storage overhead for the main database: " << overhead
              << std::endl;
  }

  Database(const std::string &mesocluster_size_file,
           const std::string &mesocluster_centroids_file,
           const std::string &index_file, index_t dim_last) {
    parse_num_clusters_per_mesocluster(mesocluster_size_file);
    parse_mesocluster_centroids(mesocluster_centroids_file);
    parse_index(index_file);
    prepare_ann_2d(dim_last);
  }

  Database(const std::string &mesocluster_size_file,
           const std::string &mesocluster_centroids_file,
           const std::string &index_file, index_t dim_mid, index_t dim_last) {
    parse_num_clusters_per_mesocluster(mesocluster_size_file);
    parse_mesocluster_centroids(mesocluster_centroids_file);
    parse_index(index_file);
    prepare_ann_3d(dim_mid, dim_last);
  }
};

} // namespace lwe_ann