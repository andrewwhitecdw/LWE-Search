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
 
#include "gemm.h"
#include "type_utils.h"

#include <limits>
#include <stdexcept>

#include <cute/atom/mma_traits.hpp>
#include <cute/tensor.hpp>

namespace lwe_ann {
namespace detail {

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

// Custom MMA handles 8x8x16 gemm with uint32_t A and int8_t B and int32_t C
// Based on mma_sm80.hpp and mma_traits_sm80.hpp
struct CustomMMA {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &a2, uint32_t const &a3,
                                   uint32_t const &b0, uint32_t const &c0,
                                   uint32_t const &c1) {
    // Beware that d0 and d1 may be equal to other registers
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    // static uint2 d_tmp[4];
    asm volatile("{\n"
                 ".reg .s32 d01, d02, d03, d11, d12, d13;\n"
                 "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
                 "{%0, %1},"
                 "{%2},"
                 "{%6},"
                 "{%7, %8};\n"
                 "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
                 "{d01, d11},"
                 "{%3},"
                 "{%6},"
                 "{0, 0};\n"
                 "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
                 "{d02, d12},"
                 "{%4},"
                 "{%6},"
                 "{0, 0};\n"
                 "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
                 "{d03, d13},"
                 "{%5},"
                 "{%6},"
                 "{0, 0};\n"
                 "shl.b32 d01, d01, 8;\n"
                 "shl.b32 d11, d11, 8;\n"
                 "shl.b32 d02, d02, 16;\n"
                 "shl.b32 d12, d12, 16;\n"
                 "shl.b32 d03, d03, 24;\n"
                 "shl.b32 d13, d13, 24;\n"
                 "add.s32 %0, %0, d01;\n"
                 "add.s32 %1, %1, d11;\n"
                 "add.s32 %0, %0, d02;\n"
                 "add.s32 %1, %1, d12;\n"
                 "add.s32 %0, %0, d03;\n"
                 "add.s32 %1, %1, d13;\n"
                 "}\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(c0),
                   "r"(c1));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use CustomMMA "
                              "without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

// Custom load from smem to registers
struct CustomCopy {
  using SRegisters = cute::uint128_t[1];
  using DRegisters = cute::uint128_t[1];

  CUTE_HOST_DEVICE static void copy(cute::uint128_t const &smem_src,
                                    cute::uint128_t &dst) {
    // load 128 bits (aligned)
    dst = smem_src;

    // The following is only true for little endian systems
#if defined(__CUDA_ARCH__)

    // The following is the same as:
    // swap 0 <-> 2 & 1 <-> 3
    // tmp.x = (dst.x & 0x0000FFFF) | ((dst.z & 0x0000FFFF) << 16);
    // tmp.z = (dst.z & 0xFFFF0000) | ((dst.x & 0xFFFF0000) >> 16);
    // tmp.y = (dst.y & 0x0000FFFF) | ((dst.w & 0x0000FFFF) << 16);
    // tmp.w = (dst.w & 0xFFFF0000) | ((dst.y & 0xFFFF0000) >> 16);
    // swap 0 <-> 1 & 2 <-> 3
    // tmp.x = (tmp.x & 0x00FF00FF) | ((tmp.y & 0x00FF00FF) << 8);
    // tmp.y = (tmp.y & 0xFF00FF00) | ((tmp.x & 0xFF00FF00) >> 8);
    // tmp.z = (tmp.z & 0x00FF00FF) | ((tmp.w & 0x00FF00FF) << 8);
    // tmp.w = (tmp.w & 0xFF00FF00) | ((tmp.z & 0xFF00FF00) >> 8);
    asm volatile("{\n"
                 ".reg .u32 tmp0, tmp1, tmp2, tmp3;\n"
                 "prmt.b32 tmp0, %0, %2, 0x5410;\n"
                 "prmt.b32 tmp2, %0, %2, 0x7632;\n"
                 "prmt.b32 tmp1, %1, %3, 0x5410;\n"
                 "prmt.b32 tmp3, %1, %3, 0x7632;\n"
                 "prmt.b32 %0, tmp0, tmp1, 0x6240;\n"
                 "prmt.b32 %1, tmp0, tmp1, 0x7351;\n"
                 "prmt.b32 %2, tmp2, tmp3, 0x6240;\n"
                 "prmt.b32 %3, tmp2, tmp3, 0x7351;\n"
                 "}\n"
                 : "+r"(reinterpret_cast<uint4 *>(&dst)->x),
                   "+r"(reinterpret_cast<uint4 *>(&dst)->y),
                   "+r"(reinterpret_cast<uint4 *>(&dst)->z),
                   "+r"(reinterpret_cast<uint4 *>(&dst)->w));
#else
    {
      struct __builtin_align__(16) uint4 {
        uint32_t x, y, z, w;
      };
      uint4 *dst_ptr = reinterpret_cast<uint4 *>(&dst);

      uint32_t tmp[4];
      constexpr uint32_t mask0011 = 0x0000FFFF;
      constexpr uint32_t mask1100 = 0xFFFF0000;
      constexpr uint32_t mask0101 = 0x00FF00FF;
      constexpr uint32_t mask1010 = 0xFF00FF00;

      tmp[0] = (dst_ptr->x & mask0011) | ((dst_ptr->z & mask0011) << 16);
      tmp[2] = (dst_ptr->z & mask1100) | ((dst_ptr->x & mask1100) >> 16);
      tmp[1] = (dst_ptr->y & mask0011) | ((dst_ptr->w & mask0011) << 16);
      tmp[3] = (dst_ptr->w & mask1100) | ((dst_ptr->y & mask1100) >> 16);

      dst_ptr->x = (tmp[0] & mask0101) | ((tmp[1] & mask0101) << 8);
      dst_ptr->y = (tmp[1] & mask1010) | ((tmp[0] & mask1010) >> 8);
      dst_ptr->z = (tmp[2] & mask0101) | ((tmp[3] & mask0101) << 8);
      dst_ptr->w = (tmp[3] & mask1010) | ((tmp[2] & mask1010) >> 8);
    }
#endif
  }
};

} // namespace detail
} // namespace lwe_ann

// The following is a structure that needs to be used in cute files

template <> struct cute::MMA_Traits<lwe_ann::detail::CustomMMA> {
  using ValTypeD = int32_t;
  using ValTypeA = uint32_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8, _8, _16>;
  using ThrID = Layout<_32>; // Warp-level atom

  using ALayout = Layout<Shape<Shape<_4, _8>, _4>, Stride<Stride<_32, _1>, _8>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, _4>, Stride<Stride<_32, _1>, _8>>;
  // (T32,V2) -> (M8,N8)
  // using SM80_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
  //                            Stride<Stride<_16,_1>,_8>>;
  using CLayout = Layout<Shape<Shape<_4, _8>, _2>, Stride<Stride<_16, _1>, _8>>;
};

// Same as UniversalCopy<uint128_t>
template <> struct cute::Copy_Traits<lwe_ann::detail::CustomCopy> {
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, _128>>;
  using DstLayout =
      Layout<Shape<_1, _128>>; // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

namespace lwe_ann {
namespace detail {

// This implementation is very naive.
// However, for small batch sizes, it should be actually faster than
// sophisticated implementations.

// LHS: M x K (K-major)
// DB: K x N (K-major)
// res: M x N (N-major)

//
// The GEMM kernel is not modified from the tutorial version.
//
template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class TiledCopyA, class S2RAtomA, class TB,
          class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class TiledMma>
__global__ static void __launch_bounds__(decltype(size(TiledMma{}))::value)
    __gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TA const *A,
                  AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                  S2RAtomA s2r_atom_a, TB const *B, BStride dB,
                  BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
                  TC *C, CStride dC, TiledMma mma,
                  const int round_shift_bits = 0) {
  using namespace cute;

  // Preconditions
  // shape_MNK == (M, N, K)
  // cta_tiler == (bM, bN, bK)
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (bM, bN, bK)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // bM
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // bN
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // bK
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // bK

  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape (M, K)
  CUTE_STATIC_ASSERT_V(
      congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape (N, K)
  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape (M, N)

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  //
  // Partition the copying of A and B tiles across the threads
  //

  // Get a partition that this thread will take care of
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) {
      ++k_tile_next;
    }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

  // tCrC --> natural ordering of _2 x _8 x (_4 x _2)

  CUTE_STATIC_ASSERT_V(
      (shape(tCrC) == take<0, 3>(shape(tCgC))));          // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA))); // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB))); // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
  Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX - 2>();
    // cp_async_wait<0>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using
  // SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and
  //     compute on rmem. Copy from smem->rmem can overlap with compute on
  //     rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_, _, _, smem_pipe_read);
        tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX - 2>();
        // cp_async_wait<0>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
      copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0) {
        copy(copy_a, tAgA(_, _, _, k_tile_next),
             tAsA(_, _, _, smem_pipe_write));
        copy(copy_b, tBgB(_, _, _, k_tile_next),
             tBsB(_, _, _, smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) {
          ++k_tile_next;
        }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read =
            (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
  }

  //
  // Epilogue
  //
  if (round_shift_bits > 0) {

    TC half_th = ((TC)1) << (round_shift_bits - 1);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      tCgC(i) = (tCrC(i) + half_th) >> round_shift_bits;
    }
  } else {
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      tCgC(i) = tCrC(i);
    }
  }
}

template <int tile_m, int tile_n>
cudastf::cuda_kernel_desc __gemm_factory(int m, int n, int k, uint32_t const *A,
                                         int8_t const *B, int32_t *C,
                                         int round_shift_bits = 0) {
  using namespace cute;

  // Keep in mind that the default setting for a layout is column-major format
  // e.g., (X, Y, Z) : (1, X, X * Y)
  // mapping data(i, j, k) -> data + i + j * X + k * X * Y

  using TA = uint32_t;
  using TB = int8_t;
  using TC = int32_t;

  if (round_shift_bits < 0 || round_shift_bits >= sizeof(TC) * 8) {
    throw std::invalid_argument("round_shift_bits must be between 0 and " +
                                std::to_string(sizeof(TC) * 8 - 1));
  }

  // Only these input shapes are dynamically determined
  // Define shapes (dynamic)
  auto M = int64_t(m);
  auto N = int64_t(n);
  auto K = int64_t(k);
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(K, Int<1>{}); // (K, 1)
  auto dB = make_stride(K, Int<1>{}); // (K, 1)
  auto dC = make_stride(N, Int<1>{}); // (N, 1)

  // Define CTA tile sizes (static)

  // It's generatlly the best to set bM * sizeof(TA) == bN * sizeof(TB)
  // This is because the arithmetric intensity is determined as
  // O(1/(bM * sizeof(TA)) + 1/(bN * sizeof(TB)))

  constexpr auto bM = Int<tile_m>{};
  constexpr auto bN = Int<tile_n>{};
  // templatizing bK fails for some reason...
  // Maybe it makes type deduction too complex?
  constexpr auto bK = Int<32>{};

  constexpr auto cta_tiler = make_shape(bM, bN, bK); // (bM, bN, bK)
  constexpr auto bP = Int<3>{};                      // Pipeline
  constexpr auto num_threads = Int<128>{}.value;

  // Each block will be in charge of creating a
  // bM x bN submatrix of C
  // which requires...
  // bM x K strip of A
  // bN x K strip of B
  constexpr auto num_TA_per_128b = sizeof(uint128_t) / sizeof(TA);
  constexpr auto num_TB_per_128b = sizeof(uint128_t) / sizeof(TB);
  static_assert(bK.value % num_TA_per_128b == 0,
                "bK must be divisible by num_TA_per_128b");
  static_assert(bK.value % num_TB_per_128b == 0,
                "bK must be divisible by num_TB_per_128b");
  constexpr auto mma_atom_shape = MMA_Traits<CustomMMA>::Shape_MNK{};
  constexpr auto atom_M = get<0>(mma_atom_shape);
  constexpr auto atom_N = get<1>(mma_atom_shape);
  constexpr auto atom_K = get<2>(mma_atom_shape);
  static_assert(bM.value % atom_M.value == 0, "bM must be divisible by atom_M");
  static_assert(bN.value % atom_N.value == 0, "bN must be divisible by atom_N");
  static_assert(bK.value % atom_K.value == 0, "bK must be divisible by atom_K");

  // smem layout is optimized for both
  // 1. bK-word-granular loads (for TA and TB)
  // 2. 128-bit-granular loads
  // This can be simply done if the following conditions are met:
  static_assert(bK.value >= num_TA_per_128b,
                "bK must be greater than or equal to num_TA_per_128b");
  static_assert(bK.value >= num_TB_per_128b,
                "bK must be greater than or equal to num_TB_per_128b");

  constexpr auto bK_div_atom_K = bK.value / atom_K.value;
  constexpr auto smem_block_size = atom_K.value * atom_M.value;

  // The following layout fills atom_M x bK matrix (or atom_N x bK matrix)
  // into smem.

  // For example,if atom_M = 8, atom_K = 16, bK = 32,
  // and we were to copy a 8 x 32 matrix
  // [  0,   1,   2, ...,  31]
  // [ 32,  33,  34, ...,  63]
  // [ 64,  65,  66, ...,  95]
  // [ 96,  97,  98, ..., 127]
  // [128, 129, 130, ..., 159]
  // [160, 161, 162, ..., 191]
  // [192, 193, 194, ..., 223]
  // [224, 225, 226, ..., 255]
  // (row-major) into smem, it will be filled in smem as follows:
  // [  0- 15] [ 32- 47] [ 64- 79] [ 96-111] [128-143] [160-175] [192-207]
  // [224-239]
  // [ 16- 31] [ 48- 63] [ 80- 95] [112-127] [144-159] [176-191] [208-223]
  // [240-255]

  // They correspond to the following smem banks:
  // B[ 0- 3] B[ 4- 7] B[ 8-11] B[12-15] B[16-19] B[20-23] B[24-27] B[28-31]
  // B[ 0- 3] B[ 4- 7] B[ 8-11] B[12-15] B[16-19] B[20-23] B[24-27] B[28-31]

  // For example, we can apply Swizzle<1, 6> in this case, which leads to:
  // data to be filled in smem like:
  // [  0- 15] [ 32- 47] [ 64- 79] [ 96-111] [128-143] [160-175] [192-207]
  // [224-239]
  // [144-159] [176-191] [208-223] [240-255] [ 16- 31] [ 48- 63] [ 80- 95]
  // [112-127]

  // Then, we can now enjoy fully utilizing the smem bandwidth with
  // interleaved banks.

  using smem_baseline_shape_A =
      Shape<decltype(atom_M), Shape<decltype(atom_K), Int<bK_div_atom_K>>>;
  using smem_baseline_stride_A =
      Stride<decltype(atom_K), Stride<_1, Int<smem_block_size>>>;
  using smem_baseline_shape_B =
      Shape<decltype(atom_N), Shape<decltype(atom_K), Int<bK_div_atom_K>>>;
  using smem_baseline_stride_B =
      Stride<decltype(atom_K), Stride<_1, Int<smem_block_size>>>;

  constexpr auto smem_baseline_layout_A =
      Layout<smem_baseline_shape_A, smem_baseline_stride_A>{};
  constexpr auto smem_baseline_layout_B =
      Layout<smem_baseline_shape_B, smem_baseline_stride_B>{};
  constexpr auto swizzle_atom_A =
      composition(Swizzle<1, 4>{}, smem_baseline_layout_A);
  constexpr auto swizzle_atom_B =
      composition(Swizzle<1, 6>{}, smem_baseline_layout_B);

  // The same data layout is used to tile the entire smem space.
  constexpr auto sA = tile_to_shape(swizzle_atom_A, make_shape(bM, bK, bP));
  constexpr auto sB = tile_to_shape(swizzle_atom_B, make_shape(bN, bK, bP));

  // Used for global memory to smem transfer
  // Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t> is a simple 128b loader
  // per thread we use one dimensional threading (only threadIdx.x is used)
  constexpr auto copy_a_k_threads = bK.value / num_TA_per_128b;
  static_assert(num_threads % copy_a_k_threads == 0,
                "num_threads must be divisible by copy_a_k_threads");
  constexpr auto copy_a_m_threads = num_threads / copy_a_k_threads;
  TiledCopy copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
      Layout<Shape<Int<copy_a_m_threads>, Int<copy_a_k_threads>>,
             Stride<Int<copy_a_k_threads>, _1>>{}, // (8, 16) : (16, 1)
      Layout<Shape<_1, Int<num_TA_per_128b>>>{});  // (1, 4) : (0, 1)
  constexpr auto copy_b_k_threads = bK.value / num_TB_per_128b;
  static_assert(num_threads % copy_b_k_threads == 0,
                "num_threads must be divisible by copy_b_k_threads");
  constexpr auto copy_b_m_threads = num_threads / copy_b_k_threads;
  TiledCopy copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
      Layout<Shape<Int<copy_b_m_threads>, Int<copy_b_k_threads>>,
             Stride<Int<copy_b_k_threads>, _1>>{}, // (32, 4) : (4, 1)
      Layout<Shape<_1, Int<num_TB_per_128b>>>{});  // (1, 16) : (0, 1)

  constexpr TiledMMA mmaC =
      make_tiled_mma(CustomMMA{}, Layout<Shape<_2, _2>>{}, // 2x2x1 MMA Atoms
                     Tile<_16, _64, _16>{});               // 16x64x16 Tiled MMA

  // Still we want to load four elements at a time, which matches with
  // s2r_atom_B;
  constexpr Copy_Atom<CustomCopy, TA> s2r_atom_A;
  constexpr Copy_Atom<SM75_U32x4_LDSM_N, TB> s2r_atom_B;

  constexpr auto smem_size =
      sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
  constexpr dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  auto kernel_fptr =
      __gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(dA),
                    decltype(sA), decltype(copyA), decltype(s2r_atom_A), TB,
                    decltype(dB), decltype(sB), decltype(copyB),
                    decltype(s2r_atom_B), TC, decltype(dC), decltype(mmaC)>;

  cudastf::cuda_safe_call(cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  cudastf::cuda_safe_call(cudaFuncSetAttribute(
      kernel_fptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

  return cudastf::cuda_kernel_desc{
      kernel_fptr, dimGrid, dimBlock, smem_size, prob_shape,
      cta_tiler,   A,       dA,       sA,        copyA,
      s2r_atom_A,  B,       dB,       sB,        copyB,
      s2r_atom_B,  C,       dC,       mmaC,      round_shift_bits};
  // kernel_fptr<<<dimGrid, dimBlock, smem_size>>>(
  //     prob_shape, cta_tiler, A, dA, sA, copyA, s2r_atom_A, B, dB, sB,
  //     copyB, s2r_atom_B, C, dC, mmaC);
}

} // namespace detail

cudastf::cuda_kernel_desc get_gemm_kernel_desc(size_t m, size_t n, size_t k,
                                               int32_t const *A,
                                               int8_t const *B, int32_t *C,
                                               size_t round_shift_bits) {
  size_t m_tile = get_gemm_m_tile_size(m);
  size_t n_tile = m_tile * 4;

  uint32_t const *A_cast = reinterpret_cast<uint32_t const *>(A);
  if (m > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("M is too large");
  }
  if (n > n_tile * 65535 || n > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("N is too large");
  }
  if (k > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("K is too large");
  }
  int m_int = static_cast<int>(m);
  int n_int = static_cast<int>(n);
  int k_int = static_cast<int>(k);
  int shift_int = static_cast<int>(round_shift_bits);

  switch (m_tile) {
  case 16:
    return detail::__gemm_factory<16, 64>(m_int, n_int, k_int, A_cast, B, C,
                                          shift_int);
  case 32:
    return detail::__gemm_factory<32, 128>(m_int, n_int, k_int, A_cast, B, C,
                                           shift_int);
  case 64:
    return detail::__gemm_factory<64, 256>(m_int, n_int, k_int, A_cast, B, C,
                                           shift_int);
  default:
    throw std::runtime_error("Unsupported m_tile size");
  }

  // This is a fallback, should never be reached
  return detail::__gemm_factory<16, 64>(m_int, n_int, k_int, A_cast, B, C,
                                        shift_int);
}

} // namespace lwe_ann
