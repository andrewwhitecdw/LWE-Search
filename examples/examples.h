/* 
 * This file contains modifications from the original work, which was 
 * licensed under the MIT License. The original copyright notice and 
 * permission notice are retained below. 
 * The modifications to this file are licensed under the Apache License, 
 * Version 2.0. 
 * 
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
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

#include "seal/seal.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <exception>

/*
Helper function: Prints the name of the example in a fancy banner.
*/
inline void print_example_banner(std::string title)
{
    if (!title.empty())
    {
        std::size_t title_length = title.length();
        std::size_t banner_length = title_length + 2 * 10;
        std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
        std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

        std::cout << std::endl << banner_top << std::endl << banner_middle << std::endl << banner_top << std::endl;
    }
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    std::cout << " coeff_modulus:" << std::endl;
    std::cout << std::hex;

    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << "0x" << coeff_modulus[i].value() << std::endl;
    }
    std::cout << "0x" << coeff_modulus.back().value() << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;

    std::cout.copyfmt(oldState);
}

/*
Helper function: Prints the `parms_id' to std::ostream.
*/
inline std::ostream &operator<<(std::ostream &stream, seal::parms_id_type parms_id)
{
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    stream << std::hex << std::setfill('0') << std::setw(16) << parms_id[0] << " " << std::setw(16) << parms_id[1]
           << " " << std::setw(16) << parms_id[2] << " " << std::setw(16) << parms_id[3] << " ";

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);

    return stream;
}

/*
Helper function: Prints a vector of floating-point values.
*/
template <typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4, int prec = 3)
{
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size)
    {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++)
        {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else
    {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++)
        {
            std::cout << " " << vec[i] << ",";
        }
        if (vec.size() > 2 * print_size)
        {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++)
        {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

/*
Helper function: Prints a matrix of values.
*/
template <typename T>
inline void print_matrix(std::vector<T> matrix, std::size_t row_size)
{
    /*
    We're not going to print every column of the matrix (there are 2048). Instead
    print this many slots from beginning and end of the matrix.
    */
    std::size_t print_size = 5;

    std::cout << std::endl;
    std::cout << "    [";
    for (std::size_t i = 0; i < print_size; i++)
    {
        std::cout << std::setw(3) << std::right << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = row_size - print_size; i < row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    std::cout << "    [";
    for (std::size_t i = row_size; i < row_size + print_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    std::cout << std::endl;
}

/*
Helper function: Print line number.
*/
inline void print_line(int line_number)
{
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

/*
Helper function: Convert a value into a hexadecimal string, e.g., uint64_t(17) --> "11".
*/
inline std::string uint64_to_hex_string(std::uint64_t value)
{
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

std::string extractTestName(const char* arg) {
    std::string fullPath(arg);
    // Find the last occurrence of a path separator
    size_t pos = fullPath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return fullPath.substr(pos + 1); // Extract the substring after the last path separator
    } else {
        return fullPath; // No path separator found, return the full string
    }
}

void printTestResults(const std::string& testName, bool testPassed, double submission_time, double execution_time) {
    fprintf(stderr, "[%s] Submission: %g\n", testName.c_str(), submission_time);
    fprintf(stderr, "[%s] Execution: %g\n", testName.c_str(), execution_time);
    const char* testStatus = testPassed ? "PASS" : "FAIL";
    fprintf(stderr, "[%s] TEST %s\n", testName.c_str(), testStatus);
}

void writeTestResultToCSV(bool testPassed, int argc, char **argv, double submission_time, double execution_time) {
    std::string testName = extractTestName(argv[0]);
    std::string testResult = testPassed ? "PASS" : "FAIL";
    
    std::ostringstream csvLine;
    csvLine << testResult << "," << testName << ",";
    for (int i = 1; i < argc; i++) {
        csvLine << argv[i] << ",";
    }
    
    // Append the performance metrics to the CSV line
    csvLine << submission_time << "s,";
    csvLine << execution_time << "s,";

    std::string csvLineStr = csvLine.str();
    if (!csvLineStr.empty()) csvLineStr.pop_back();  // Remove the last comma
    csvLineStr += "\n";

    try {
        std::ofstream file("results.csv", std::ios::app);
        if (!file) {
            throw std::runtime_error("Unable to open file.");
        }
        file << csvLineStr;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

inline size_t customHash(size_t value) {
    size_t hash = value;
    
    // Mix the bits using XOR and bit shifts
    hash ^= (hash << 13);
    hash ^= (hash >> 17);
    hash ^= (hash << 5);
    
    return hash;
}

inline void loop_dispatch(size_t num_threads, size_t num_devices, size_t start, size_t end, std::function<void(size_t)> func) {
    size_t actual_num_threads = std::min(num_threads, end - start);
    std::vector<std::thread> threads;

    for (size_t tid = 0; tid < actual_num_threads; ++tid) {
        threads.emplace_back([=, &func]() {
            // Set the CUDA device for this thread
            cudaError_t cudaStatus = cudaSetDevice(tid % num_devices);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                return;
            }

	    size_t cnt = end - start + 1;
	    size_t block_size = (cnt + actual_num_threads - 1)/actual_num_threads;

	    for (size_t i = start; i < end; i++)
            {
               // std::hash<size_t> hasher;
                if (customHash(i) % actual_num_threads == tid) {
                    func(i);
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

namespace seal {

// Function to get the output directory
std::string get_output_directory() {
    const char* output_dir = std::getenv("SEAL_DUMP_DIR");
    return (output_dir != nullptr) ? std::string(output_dir) : "outputs";
}

// Function to get the compare directory
std::string get_compare_directory() {
    const char* compare_dir = std::getenv("SEAL_COMPARE_DIR");
    return (compare_dir != nullptr) ? std::string(compare_dir) : "";
}

// Function to get the comparison tolerance
double get_compare_tolerance() {
    const char* compare_tolerance = std::getenv("SEAL_COMPARE_TOLERANCE");
    return (compare_tolerance != nullptr) ? std::stod(compare_tolerance) : 0.0;
}

// Function to compare two vectors with a tolerance
template <typename R>
bool compare_vectors(const std::vector<R>& vec1, const std::vector<R>& vec2, double tolerance) {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) return false;
    }
    return true;
}

// Function to generate a default file name with an incrementing counter
inline std::string generate_default_filename() {
    static int counter = 0;

    std::ostringstream oss;
    oss << "output_" << counter++ << ".txt";

    return oss.str();
}


template <typename R>
inline void dump_ctxt(Decryptor &decryptor , CKKSEncoder& encoder,  Ciphertext &ctxt, const std::optional<std::string> &outname = std::nullopt) {
    // Decrypt and decode the result
    Plaintext plain_result;
    decryptor.decrypt(ctxt, plain_result);
    std::vector<R> result;
    encoder.decode(plain_result, result);

    // Get the output directory
    std::string output_directory = get_output_directory();

    // Ensure the directory exists
    std::filesystem::create_directories(output_directory);

    // Determine the file name
    std::string filename = outname.has_value() ? outname.value() : generate_default_filename();
    std::string full_filename = output_directory + "/" + filename;

    // Open the file for writing
    std::ofstream outfile(full_filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << full_filename << std::endl;
        return;
    }

    // Print the contents of the vector to the file
    for (const auto &value : result) {
        outfile << value << "\n";
    }

    // Close the file
    outfile.close();

    // Check for comparison directory and compare if exists
    std::string compare_directory = get_compare_directory();
    if (!compare_directory.empty()) {
        std::string compare_filename = compare_directory + "/" + filename;
        std::ifstream compare_file(compare_filename);
        if (compare_file.is_open()) {
            std::vector<R> compare_result;
            R value;
            while (compare_file >> value) {
                compare_result.push_back(value);
            }
            compare_file.close();

            double tolerance = get_compare_tolerance();
            if (!compare_vectors(result, compare_result, tolerance)) {
                std::cerr << "Comparison failed for file: " << full_filename << std::endl;
            } else {
                std::cout << "Comparison succeeded for file: " << full_filename << std::endl;
            }
        } else {
            std::cerr << "Failed to open comparison file: " << compare_filename << std::endl;
        }
    }
}
} // end namespace seal
