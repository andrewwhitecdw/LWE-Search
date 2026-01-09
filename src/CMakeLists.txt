# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/lwe_server.cu
    ${CMAKE_CURRENT_LIST_DIR}/rlwe_server.cu
    ${CMAKE_CURRENT_LIST_DIR}/gemm.cu
    ${CMAKE_CURRENT_LIST_DIR}/simplepir.cu
    ${CMAKE_CURRENT_LIST_DIR}/rlwe_compress.cu
)

install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/arith_utils.h
        ${CMAKE_CURRENT_LIST_DIR}/type_utils.h
        ${CMAKE_CURRENT_LIST_DIR}/lwe_cudastf.h
        ${CMAKE_CURRENT_LIST_DIR}/lwe_client.h
        ${CMAKE_CURRENT_LIST_DIR}/lwe_server.h
        ${CMAKE_CURRENT_LIST_DIR}/rlwe_client.h
        ${CMAKE_CURRENT_LIST_DIR}/rlwe_server.h
        ${CMAKE_CURRENT_LIST_DIR}/gemm.h
        ${CMAKE_CURRENT_LIST_DIR}/simplepir.h
        ${CMAKE_CURRENT_LIST_DIR}/rlwe_compress.h
        ${CMAKE_CURRENT_LIST_DIR}/database.h
        ${CMAKE_CURRENT_LIST_DIR}/reconstruct_words.h
    DESTINATION
        ${SEAL_INCLUDES_INSTALL_DIR}/seal/ann_kernels
)

set(CUDA_FILES lwe_server.cu simplepir.cu gemm.cu rlwe_server.cu)
set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES} PARENT_SCOPE)
