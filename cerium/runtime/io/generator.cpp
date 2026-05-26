// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#include "cerium/runtime/io/generator.h"
#include "cerium/runtime/utils/overloaded.h"
#include "cerium/runtime/utils/string.h"

namespace Cerium {
namespace Runtime {

namespace {

constexpr size_t kPlaintextsFileBufferSize = 8 * 1024 * 1024;

class BufferedPlaintextsFile {
public:
  explicit BufferedPlaintextsFile(const std::string &file_name)
      : file_name_(file_name), buffer_(kPlaintextsFileBufferSize),
        stream_(&file_) {
    file_.pubsetbuf(buffer_.data(), buffer_.size());
    file_.open(file_name, std::ios::in | std::ios::binary);
  }

  ~BufferedPlaintextsFile() { close(); }

  bool is_open() const { return file_.is_open(); }

  std::istream &read(char *data, std::streamsize size) {
    return stream_.read(data, size);
  }

  void close() {
    if (!file_.is_open()) {
      return;
    }

    file_.close();

    // The plaintext has been fully deserialized.  Do not retain it in the
    // page cache: the next phase allocates large UVM host regions and benefits
    // from immediately reclaimable normal pages.
    const int fd = ::open(file_name_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
      (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      (void)::close(fd);
    }
  }

private:
  std::string file_name_;
  std::vector<char> buffer_;
  std::filebuf file_;
  std::istream stream_;
};

struct RuntimeInputRow {
  std::string term;
  std::string descriptor;
  std::vector<uint32_t> rns_base_ids;
};

std::string trim_copy(const std::string &value) {
  const auto begin = value.find_first_not_of(" \t\n\r\f\v");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\n\r\f\v");
  return value.substr(begin, end - begin + 1);
}

std::vector<uint32_t> parse_uint32_list(const std::string &list,
                                        const std::string &context) {
  auto values = trim_copy(list);
  if (values.size() < 2 || values.front() != '[' || values.back() != ']') {
    throw std::runtime_error("Invalid bracketed list in " + context + ": " +
                             list);
  }

  values = values.substr(1, values.size() - 2);
  std::vector<uint32_t> result;
  size_t start = 0;
  while (start <= values.size()) {
    const auto end = values.find(',', start);
    auto token = trim_copy(values.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (!token.empty()) {
      const auto value = std::stoul(token);
      if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("RNS base id exceeds uint32_t in " + context +
                                ": " + token);
      }
      result.push_back(static_cast<uint32_t>(value));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

struct RuntimeIOEntry {
  std::string term;
  std::string var;
  std::string type;
  std::vector<uint32_t> rns_base_ids;
};

RuntimeInputRow parse_runtime_input_row(const std::string &line) {
  const auto first_separator = line.find('|');
  if (first_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input row: " + line);
  }

  const auto second_separator = line.find('|', first_separator + 1);
  if (second_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input row: " + line);
  }

  RuntimeInputRow row;
  row.term = trim_copy(line.substr(0, first_separator));
  row.descriptor = trim_copy(
      line.substr(first_separator + 1, second_separator - first_separator - 1));
  row.rns_base_ids = parse_uint32_list(line.substr(second_separator + 1), line);
  return row;
}

RuntimeIOEntry parse_runtime_io(const std::string &line) {
  auto row = parse_runtime_input_row(line);
  const auto descriptor_separator = row.descriptor.find(':');
  if (descriptor_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input descriptor: " +
                             row.descriptor);
  }

  RuntimeIOEntry entry;
  entry.term = std::move(row.term);
  entry.var = row.descriptor.substr(0, descriptor_separator);
  entry.type = row.descriptor.substr(descriptor_separator + 1);
  entry.rns_base_ids = std::move(row.rns_base_ids);
  return entry;
}

} // namespace

IOGenerator::EvalkeyInfo::EvalkeyInfo(std::string &&key_info,
                                      std::string &&rns_bases_str) {

  std::vector<std::string> split = Cerium::Runtime::Utils::split_string(key_info, ":");
  if (split.size() != 7 || split[0] != "K") {
    throw std::runtime_error("Invalid Evalkey Entry" + key_info);
  }

  auto evk_type = std::stoi(split[1]);
  switch (evk_type) {
  case 0:
    key_type = KeyType::Mul;
    break;
  case 1:
    key_type = KeyType::Rot;
    break;
  case 2:
    key_type = KeyType::Con;
    break;
  case 3:
    key_type = KeyType::Boot;
    break;
  case 4:
    key_type = KeyType::Ephemeral;
    break;
  case 5:
    key_type = KeyType::Boot2;
    break;
  case 6:
    key_type = KeyType::RotInv;
    break;
  default:
    throw std::runtime_error("Invalid Evalkey type: " + split[1]);
  }

  id = split[1] + ":" + split[2] + ":" + split[3] + ":" + split[4] + ":" +
       split[6];

  level = std::stoi(split[2]);
  extension_size = std::stoi(split[3]);
  rotation_amount = std::stoi(split[4]);

  split[5][0] = '0';
  ct_number = std::stoi(split[5]);

  auto &digit_partition_str = split[6];
  digit_partition_str[0] = ' ';
  digit_partition_str[digit_partition_str.length() - 1] = ' ';

  auto digit_partition_str_split =
      Cerium::Runtime::Utils::split_string(digit_partition_str, ",");
  for (auto &i : digit_partition_str_split) {
    digit_partition.push_back(std::stoul(i));
  }

  rns_bases_str[0] = ' ';
  rns_bases_str[rns_bases_str.length() - 1] = ' ';

  auto rns_bases_str_split = Cerium::Runtime::Utils::split_string(rns_bases_str, ",");
  for (auto &i : rns_bases_str_split) {
    rns_bases.push_back(std::stoul(i));
  }
}

std::pair<std::string, IOGenerator::EvalKeyEntryType>
IOGenerator::parse_evalkey(std::string &line) {

  line.erase(
      std::remove_if(line.begin(), line.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      line.end());
  auto split = Cerium::Runtime::Utils::split_string(line, "|");
  if (split.size() != 3) {
    throw std::runtime_error("Invalid line: " + line);
  }

  auto &term = split[0];
  auto &var_type = split[1];
  auto &rns_bases = split[2];

  EvalkeyInfo evk_info(std::move(var_type), std::move(rns_bases));
  return std::pair(term, std::pair(evk_info, std::pair(nullptr, nullptr)));
};

IOGenerator::PlaintextInfo::PlaintextInfo(std::string &&info,
                                          std::string &&rns_bases_str) {

  std::vector<std::string> split = Cerium::Runtime::Utils::split_string(info, ":");
  if (split.size() != 3 || split[1] != "p") {
    throw std::runtime_error("Invalid Plaintext Entry" + info);
  }

  var = split[0];
  elt_size = std::stoul(split[2]);

  // Ensure elt_size is a power of 2
  assert((elt_size & (elt_size - 1)) == 0);

  rns_bases_str[0] = ' ';
  rns_bases_str[rns_bases_str.length() - 1] = ' ';

  auto rns_bases_str_split = Cerium::Runtime::Utils::split_string(rns_bases_str, ",");
  for (auto &i : rns_bases_str_split) {
    rns_bases.push_back(std::stoul(i));
  }
}

std::pair<std::string, IOGenerator::PlaintextEntryType>
IOGenerator::parse_plaintext(std::string &line) {

  line.erase(
      std::remove_if(line.begin(), line.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      line.end());
  auto split = Cerium::Runtime::Utils::split_string(line, "|");
  if (split.size() != 3) {
    throw std::runtime_error("Invalid line: " + line);
  }

  auto &term = split[0];
  auto &info = split[1];
  auto &rns_bases = split[2];

  PlaintextInfo plaintext_info(std::move(info), std::move(rns_bases));
  return std::pair(term, std::pair(plaintext_info, nullptr));
};

std::unordered_map<std::string, IOGenerator::RnsPolynomialPair> IOGenerator::generate_ciphertext_inputs(
    const std::string &inputs_file_name,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
    CKKSEncryptorContext &encryptor_context) {
  auto &raw_inputs = raw_inputs_ptr->get_raw_inputs();

  LOG(logger, INFO) << "Generating Ciphertexts:" << inputs_file_name << "\n"
                    << std::flush;

  std::ifstream input_file(inputs_file_name, std::ios::in);
  std::string line;
  bool ciphertext_stream_found = false;
  while (std::getline(input_file, line)) {
    size_t pos;
    if ((pos = line.find("Ciphertext Stream")) != std::string::npos) {
      ciphertext_stream_found = true;
      break;
    }
  }

  if (!ciphertext_stream_found) {
    throw std::runtime_error("Ciphertext Stream Not Found");
  }
  bool complete = false;

  std::unordered_map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
      encrypted_inputs;

  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto encryptors = encryptor_context.create_encryptors(1);
  auto & encryptor = encryptors.at(0);
  while (std::getline(input_file, line)) {
    if (line == ";") {
      complete = true;
      break;
    }
    auto parse = parse_io(line);
    auto &term = parse.term;
    auto &var = parse.var;
    auto &type = parse.type;
    auto &rns_base_ids = parse.rns_base_ids;
    auto it = encrypted_inputs.find(var);
    if (it == encrypted_inputs.end()) {
      Cerium::Runtime::Utils::RawInputType raw_input;
      try {
        raw_input = raw_inputs.at(var);
      } catch (const std::out_of_range &err) {
        throw std::runtime_error("Raw Inputs key error: " + var);
      }
      auto &message = std::get<0>(raw_input);
      auto &scale = std::get<1>(raw_input);
      std::visit(Cerium::Runtime::Utils::overloaded{
                      [](auto arg) {
                        throw std::invalid_argument("Message" + std::string());
                      },
                      [&](const double &arg) {
                        encrypted_inputs[var] = encryptor.encode_and_encrypt(
                            arg, scale, rns_base_ids, pool_glo);
                      },
                      [&](const std::vector<double> &arg) {
                        encrypted_inputs[var] = encryptor.encode_and_encrypt(
                            arg, scale, rns_base_ids, pool_glo);
                      },
                      [&](const std::vector<std::complex<double>> &arg) {
                        encrypted_inputs[var] = encryptor.encode_and_encrypt(
                            arg, scale, rns_base_ids, pool_glo);
                      }},
                  message);
      it = encrypted_inputs.find(var);
    }
  }

  if(!complete) {
    throw std::logic_error("Unexpected EOF");
  }
  input_file.close();
  return encrypted_inputs;
}

std::pair<std::unordered_map<std::string, IOGenerator::RnsPolynomialPair>,
          size_t>
IOGenerator::generate_evalkeys(const std::string &inputs_file_name,
                               CKKSEncryptorContext &encryptor_context) {
  LOG(logger, INFO) << "Generating Evalkeys:" << inputs_file_name << "\n"
                    << std::flush;

  std::ifstream input_file(inputs_file_name, std::ios::in);
  std::string line;
  bool evalkey_stream_found = false;
  while (std::getline(input_file, line)) {
    size_t pos;
    if ((pos = line.find("Evalkey Stream")) != std::string::npos) {
      evalkey_stream_found = true;
      break;
    }
  }

  if (!evalkey_stream_found) {
    throw std::runtime_error("Evalkeys Stream Not Found");
  }

  std::unordered_map<std::string, EvalkeyInfo> evalkey_entries;
  std::unordered_map<std::string, RnsPolynomialPair> evalkeys;
  bool complete = false;
  while (std::getline(input_file, line)) {
    if (line == ";") {
      complete = true;
      break;
    }
    auto parse = parse_evalkey(line);
    auto &evk_info = std::get<0>(std::get<1>(parse));
    evalkey_entries.insert({evk_info.id, evk_info});
  }

  if (!complete) {
    throw std::logic_error("Unexpected EOF");
  }
  std::vector<std::string> evk_ids;
  evk_ids.reserve(evalkey_entries.size());

  for (auto &[k, v] : evalkey_entries) {
    evk_ids.push_back(k);
  }

  uint8_t NUM_THREADS = 64;
  std::vector<decltype(evalkeys)> thread_local_evalkeys(NUM_THREADS);
  std::vector<size_t> thread_local_evalkeys_size(NUM_THREADS, 0);

  auto encryptors = encryptor_context.create_encryptors(NUM_THREADS);

  auto process_evalkey = [&](int tid) {
    for (auto i = tid; i < evk_ids.size(); i += NUM_THREADS) {
      const auto &evk_info = evalkey_entries.at(evk_ids.at(i));

      std::pair<RnsPolynomialPtr, RnsPolynomialPtr> evk;

      auto &encryptor = encryptors.at(tid);

      switch (evk_info.key_type) {
      case EvalkeyInfo::KeyType::Mul: {
        evk = encryptor.generate_relin_evalkey(
            evk_info.level, evk_info.extension_size, evk_info.digit_partition,
            pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::Rot: {
        evk = encryptor.generate_rotation_evalkey(
            evk_info.rotation_amount, evk_info.level, evk_info.extension_size,
            evk_info.digit_partition, pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::Con: {
        evk = encryptor.generate_conjugation_evalkey(
            evk_info.level, evk_info.extension_size, evk_info.digit_partition,
            pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::Boot: {
        evk = encryptor.generate_bootstrap_evalkey(
            evk_info.level, evk_info.extension_size, evk_info.digit_partition,
            pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::Ephemeral: {
        evk = encryptor.generate_ephemeral_evalkey(
            evk_info.level, evk_info.extension_size, evk_info.digit_partition,
            pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::Boot2: {
        evk = encryptor.generate_bootstrap_evalkey2(
            evk_info.level, evk_info.extension_size, evk_info.digit_partition,
            pool_glo);
      }; break;
      case EvalkeyInfo::KeyType::RotInv: {
        evk = encryptor.generate_rotation_inv_evalkey(
            evk_info.rotation_amount, evk_info.level, evk_info.extension_size,
            evk_info.digit_partition, pool_glo);
      }; break;
      default:
        throw std::runtime_error("Invalid Evalkey Type");
      }

      std::size_t num_limbs = evk_info.level + evk_info.extension_size;
      const auto &evk0 = std::get<0>(evk);
      const auto &evk1 = std::get<1>(evk);
      assert(evk0->num_limbs() == num_limbs);
      assert(evk1->num_limbs() == num_limbs);
      thread_local_evalkeys_size[tid] += (evk0->size() + evk1->size());
      thread_local_evalkeys.at(tid)[evk_info.id] = std::move(evk);
    }
  };

  std::vector<std::thread> threads;
  for (size_t tid = 0; tid < NUM_THREADS; tid++) {
    threads.push_back(std::thread(process_evalkey, tid));
  }

  for (size_t tid = 0; tid < NUM_THREADS; tid++) {
    threads[tid].join();
  }

  size_t evalkeys_size = 0;
  for (int tid = 0; tid < NUM_THREADS; tid++) {
    auto &local_evalkeys_ = thread_local_evalkeys.at(tid);
    evalkeys.insert(local_evalkeys_.begin(), local_evalkeys_.end());
    evalkeys_size += thread_local_evalkeys_size[tid];
  }
  thread_local_evalkeys.clear();
  return std::pair(evalkeys, evalkeys_size);
}

void IOGenerator::serialize_evalkeys(
    const std::string &output_file_name,
    const std::unordered_map<std::string, IOGenerator::RnsPolynomialPair>
        &evalkeys,
    size_t evalkeys_size) {

  std::ofstream output_file(output_file_name, std::ios::out | std::ios::binary);
  std::stringstream ss;
  output_file.write(evalkey_header.c_str(), evalkey_header.size() + 1);

  std::size_t context_n = context.n();
  output_file.write(reinterpret_cast<const char *>(&context_n),
                    sizeof(context_n));
  std::size_t context_num_rns_bases = context.num_rns_bases();
  output_file.write(reinterpret_cast<const char *>(&context_num_rns_bases),
                    sizeof(context_num_rns_bases));
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < rns_bases.size(); i++) {
    uint64_t modulus = rns_bases.at(i).value();
    output_file.write(reinterpret_cast<const char *>(&modulus),
                      sizeof(modulus));
  }

  std::size_t num_evalkeys = evalkeys.size();
  output_file.write(reinterpret_cast<const char *>(&num_evalkeys),
                    sizeof(num_evalkeys));

  std::size_t evalkeys_size_bytes = evalkeys_size * sizeof(Limb::Element_t);
  output_file.write(reinterpret_cast<const char *>(&evalkeys_size_bytes),
                    sizeof(evalkeys_size_bytes));

  for (auto &[k, v] : evalkeys) {
    auto evk = evalkeys.at(k);
    std::size_t evk_entry_id_size = k.size() + 1;
    output_file.write(reinterpret_cast<const char *>(&evk_entry_id_size),
                      sizeof(evk_entry_id_size));
    output_file.write(k.c_str(), evk_entry_id_size);
    auto evk0 = std::get<0>(evk);
    auto evk1 = std::get<1>(evk);

    std::size_t num_limbs = evk0->num_limbs();
    assert(evk1->num_limbs() == evk0->num_limbs());

    output_file.write(reinterpret_cast<const char *>(&num_limbs),
                      sizeof(num_limbs));

    for (auto &[limb_id, limb] : evk0->limb_index_map()) {
      output_file.write(reinterpret_cast<const char *>(&limb_id),
                        sizeof(limb_id));
      assert(limb->size() == context_n);
      output_file.write(reinterpret_cast<const char *>(limb->data()),
                        context_n * sizeof(Limb::Element_t));
    }
    for (auto &[limb_id, limb] : evk1->limb_index_map()) {
      output_file.write(reinterpret_cast<const char *>(&limb_id),
                        sizeof(limb_id));
      assert(limb->size() == context_n);
      output_file.write(reinterpret_cast<const char *>(limb->data()),
                        context_n * sizeof(Limb::Element_t));
    }
  }
}

void IOGenerator::generate_and_serialize_evalkeys(
    const std::string &output_file_name, const std::string &inputs_file_name,
    CKKSEncryptorContext &encryptor_context) {
  auto gen = generate_evalkeys(inputs_file_name, encryptor_context);
  serialize_evalkeys(output_file_name, std::get<0>(gen), std::get<1>(gen));
};

void IOGenerator::generate_and_serialize_plaintexts(
    const std::string &output_file_name, const std::string &inputs_file_name,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
    const std::string &stream_name) {
  auto gen = generate_plaintexts(inputs_file_name, raw_inputs_ptr, stream_name);
  serialize_plaintexts(output_file_name, std::get<0>(gen), std::get<1>(gen));
};

void IOGenerator::generate_and_serialize_ciphertexts(
    const std::string &output_file_name, const std::string &inputs_file_name,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
    CKKSEncryptorContext &encryptor_context) {
  auto ciphertexts = generate_ciphertext_inputs(inputs_file_name, raw_inputs_ptr,
                                                encryptor_context);

  size_t ciphertexts_size = 0;
  for (const auto &[name, ciphertext] : ciphertexts) {
    const auto &c0 = std::get<0>(ciphertext);
    const auto &c1 = std::get<1>(ciphertext);
    if (!c0 || !c1) {
      throw std::runtime_error("Generated ciphertext has a null component: " +
                               name);
    }
    if (c0->num_limbs() != c1->num_limbs()) {
      throw std::runtime_error(
          "Ciphertext components have different limb counts: " + name);
    }
    ciphertexts_size += c0->size() + c1->size();
  }

  serialize_ciphertexts(output_file_name, ciphertexts, ciphertexts_size);
}

std::pair<std::unordered_map<std::string, RnsPolynomialPtr>, size_t>
IOGenerator::generate_plaintexts(
    const std::string &inputs_file_name,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
    const std::string &stream_name) {

  auto &raw_inputs = raw_inputs_ptr->get_raw_inputs();
  LOG(logger, INFO) << "Generating Plaintext Inputs:" << inputs_file_name
                    << "\n"
                    << std::flush;
  std::ifstream input_file(inputs_file_name, std::ios::in);
  std::string line;

  int stream_count = 0;

  bool plaintext_stream_found = false;
  while (std::getline(input_file, line)) {
    size_t pos;
    if ((pos = line.find(stream_name)) != std::string::npos) {
      plaintext_stream_found = true;
      break;
    }
  }
  if (!plaintext_stream_found) {
    throw std::runtime_error(stream_name + " Not Found");
  }

  std::unordered_map<std::string, RnsPolynomialPtr> plaintexts;
  size_t plaintexts_size = 0;

  std::vector<std::string> plaintexts_str;

  // First Check that all plaintexts have been provided
  bool complete = false;
  while (std::getline(input_file, line)) {
    if (line == ";") {
      complete = true;
      break;
    }
    auto parse = parse_plaintext(line);
    auto &plaintext_info = std::get<0>(std::get<1>(parse));
    auto &var = plaintext_info.var;
    if (raw_inputs.find(var) == raw_inputs.end()) {
      std::cerr << "ERROR: Raw Input key error: " << var << "\n" << std::flush;
      throw std::runtime_error("Raw Inputs key error: " + var);
    }
    plaintexts_str.push_back(line);
  }

  if (!complete) {
    throw std::logic_error("Unexpected EOF");
  }

  std::reverse(plaintexts_str.begin(), plaintexts_str.end());

  // int NUM_THREADS = 16;
  int NUM_THREADS = 128;
  std::vector<CKKSEncoder> encoders;
  encoders.reserve(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    encoders.emplace_back(context);
  }
  std::vector<decltype(plaintexts)> thread_local_plaintext_inputs(NUM_THREADS);
  std::vector<size_t> thread_local_plaintext_sizes(NUM_THREADS, 0);
  auto process_plaintext = [&](int tid) {
    auto &local_plaintext_inputs = thread_local_plaintext_inputs.at(tid);
    for (int i = tid; i < plaintexts_str.size(); i += NUM_THREADS) {
      auto line = plaintexts_str.at(i);
      auto parse = parse_plaintext(line);
      auto &info = std::get<0>(std::get<1>(parse));
      auto &var = info.var;
      auto &elt_size = info.elt_size;
      auto &rns_base_ids = info.rns_bases;
      auto &encoder = encoders.at(tid);
      Cerium::Runtime::Utils::RawInputType raw_input;
      try {
        raw_input = raw_inputs.at(var);
      } catch (const std::out_of_range &err) {
        std::cerr << "ERROR: Raw Input key error: " << var << "\n"
                  << std::flush;
        throw std::runtime_error("Raw Inputs key error: " + var);
      }

      auto &message = std::get<0>(raw_input);
      auto &scale = std::get<1>(raw_input);

      RnsPolynomialPtr pt;
      std::visit(Cerium::Runtime::Utils::overloaded{
                     [](auto arg) { throw std::invalid_argument("Message"); },
                     [&](const double &arg) {
                       pt = encoder.encode(arg, scale, rns_base_ids);
                       if (elt_size != context.slots()) {
                         throw std::runtime_error(
                             "Elt_size != context slot size for var: " + var);
                       }
                     },
                     [&](const std::vector<double> &arg) {
                       pt = encoder.encode(arg, scale, rns_base_ids, elt_size,
                                           true /* Perform NTT*/);
                     },
                     [&](const std::vector<std::complex<double>> &arg) {
                       pt = encoder.encode(arg, scale, rns_base_ids, elt_size,
                                           true /*Perform NTT*/);
                     }},
                 message);
      local_plaintext_inputs.insert({var, pt});
      thread_local_plaintext_sizes[tid] += pt->size();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);
  for (size_t tid = 0; tid < NUM_THREADS; tid++) {
    threads.push_back(std::thread(process_plaintext, tid));
  }

  for (size_t tid = 0; tid < NUM_THREADS; tid++) {
    threads[tid].join();
  }

  for (int tid = 0; tid < NUM_THREADS; tid++) {
    auto &local_plaintext_inputs = thread_local_plaintext_inputs.at(tid);
    plaintexts.insert(local_plaintext_inputs.begin(),
                      local_plaintext_inputs.end());
    plaintexts_size += thread_local_plaintext_sizes[tid];
  }
  thread_local_plaintext_inputs.clear();

  return std::pair(plaintexts, plaintexts_size);
}

void IOGenerator::serialize_plaintexts(
    const std::string &output_file_name,
    const std::unordered_map<std::string, RnsPolynomialPtr> &plaintexts,
    size_t plaintexts_size) {

  std::ofstream output_file(output_file_name, std::ios::out | std::ios::binary);
  std::stringstream ss;
  output_file.write(plaintext_header.c_str(), plaintext_header.size() + 1);

  std::size_t context_n = context.n();
  output_file.write(reinterpret_cast<const char *>(&context_n),
                    sizeof(context_n));
  std::size_t context_num_rns_bases = context.num_rns_bases();
  output_file.write(reinterpret_cast<const char *>(&context_num_rns_bases),
                    sizeof(context_num_rns_bases));
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < rns_bases.size(); i++) {
    uint64_t modulus = rns_bases.at(i).value();
    output_file.write(reinterpret_cast<const char *>(&modulus),
                      sizeof(modulus));
  }

  std::size_t num_plaintexts = plaintexts.size();
  output_file.write(reinterpret_cast<const char *>(&num_plaintexts),
                    sizeof(num_plaintexts));

  std::size_t plaintexts_size_bytes = plaintexts_size * sizeof(Limb::Element_t);
  output_file.write(reinterpret_cast<const char *>(&plaintexts_size_bytes),
                    sizeof(plaintexts_size_bytes));

  for (auto &[k, pt] : plaintexts) {
    std::size_t str_size = k.size() + 1;
    output_file.write(reinterpret_cast<const char *>(&str_size),
                      sizeof(str_size));
    output_file.write(k.c_str(), str_size);

    std::size_t num_limbs = pt->num_limbs();
    output_file.write(reinterpret_cast<const char *>(&num_limbs),
                      sizeof(num_limbs));

    std::size_t elt_count = pt->limb_size();
    output_file.write(reinterpret_cast<const char *>(&elt_count),
                      sizeof(elt_count));

    for (auto &[limb_id, limb] : pt->limb_index_map()) {
      output_file.write(reinterpret_cast<const char *>(&limb_id),
                        sizeof(limb_id));
      output_file.write(reinterpret_cast<const char *>(limb->data()),
                        elt_count * sizeof(Limb::Element_t));
    }
  }
  output_file.close();
}

void IOGenerator::serialize_ciphertexts(
    const std::string &output_file_name,
    const std::unordered_map<std::string, RnsPolynomialPair> &ciphertexts,
    size_t ciphertexts_size) {
  std::ofstream output_file(output_file_name, std::ios::out | std::ios::binary);
  if (!output_file.is_open()) {
    throw std::runtime_error("Ciphertexts file :" + output_file_name +
                             " cannot be opened");
  }

  output_file.write(ciphertext_header.c_str(), ciphertext_header.size() + 1);
  const std::size_t context_n = context.n();
  output_file.write(reinterpret_cast<const char *>(&context_n),
                    sizeof(context_n));
  const std::size_t context_num_rns_bases = context.num_rns_bases();
  output_file.write(reinterpret_cast<const char *>(&context_num_rns_bases),
                    sizeof(context_num_rns_bases));
  for (const auto &rns_base : context.rns_bases()) {
    const uint64_t modulus = rns_base.value();
    output_file.write(reinterpret_cast<const char *>(&modulus),
                      sizeof(modulus));
  }

  const std::size_t num_ciphertexts = ciphertexts.size();
  output_file.write(reinterpret_cast<const char *>(&num_ciphertexts),
                    sizeof(num_ciphertexts));
  const std::size_t ciphertexts_size_bytes =
      ciphertexts_size * sizeof(Limb::Element_t);
  output_file.write(reinterpret_cast<const char *>(&ciphertexts_size_bytes),
                    sizeof(ciphertexts_size_bytes));

  for (const auto &[name, ciphertext] : ciphertexts) {
    const auto &c0 = std::get<0>(ciphertext);
    const auto &c1 = std::get<1>(ciphertext);
    const std::size_t name_size = name.size() + 1;
    output_file.write(reinterpret_cast<const char *>(&name_size),
                      sizeof(name_size));
    output_file.write(name.c_str(), name_size);

    const std::size_t num_limbs = c0->num_limbs();
    output_file.write(reinterpret_cast<const char *>(&num_limbs),
                      sizeof(num_limbs));
    const std::size_t elt_count = c0->limb_size();
    if (c1->limb_size() != elt_count) {
      throw std::runtime_error(
          "Ciphertext components have different limb sizes: " + name);
    }
    output_file.write(reinterpret_cast<const char *>(&elt_count),
                      sizeof(elt_count));

    for (const auto *component : {c0.get(), c1.get()}) {
      for (const auto &[limb_id, limb] : component->limb_index_map()) {
        if (limb->size() != elt_count) {
          throw std::runtime_error("Ciphertext limb has an unexpected size: " +
                                   name);
        }
        output_file.write(reinterpret_cast<const char *>(&limb_id),
                          sizeof(limb_id));
        output_file.write(reinterpret_cast<const char *>(limb->data()),
                          elt_count * sizeof(Limb::Element_t));
      }
    }
  }

  if (!output_file) {
    throw std::runtime_error("Failed while serializing ciphertexts to: " +
                             output_file_name);
  }
}

std::unordered_map<std::string, IOGenerator::RnsPolynomialPair>
IOGenerator::deserialize_ciphertexts(
    const Context &context, const std::string &ciphertexts_file_name,
    std::function<MemoryManagerHandle(size_t size)>
        memory_pool_create_callback) {
  std::unordered_map<std::string, RnsPolynomialPair> ciphertexts;
  std::ifstream ciphertexts_file(ciphertexts_file_name,
                                 std::ios::in | std::ios::binary);
  if (!ciphertexts_file.is_open()) {
    throw std::runtime_error(
        "Ciphertexts file: " + ciphertexts_file_name +
        " cannot be opened. Did you forget to generate the ciphertexts?");
  }

  auto read_or_throw = [&](char *data, std::streamsize size) {
    ciphertexts_file.read(data, size);
    if (!ciphertexts_file) {
      throw std::runtime_error("Unexpected end of ciphertexts file: " +
                               ciphertexts_file_name);
    }
  };

  std::vector<char> header_buf(ciphertext_header.size() + 1);
  read_or_throw(header_buf.data(), header_buf.size());
  if (ciphertext_header != std::string(header_buf.data())) {
    throw std::runtime_error("Unexpected Ciphertexts Header");
  }

  std::size_t context_n = 0;
  read_or_throw(reinterpret_cast<char *>(&context_n), sizeof(context_n));
  if (context_n != context.n()) {
    throw std::runtime_error("Unexpected Context Size");
  }

  std::size_t context_num_rns_bases = 0;
  read_or_throw(reinterpret_cast<char *>(&context_num_rns_bases),
                sizeof(context_num_rns_bases));
  if (context_num_rns_bases != context.num_rns_bases()) {
    throw std::runtime_error("Unexpected Number of RNS Bases");
  }

  const auto &rns_bases = context.rns_bases();
  for (std::size_t i = 0; i < context_num_rns_bases; i++) {
    uint64_t modulus = 0;
    read_or_throw(reinterpret_cast<char *>(&modulus), sizeof(modulus));
    if (modulus != rns_bases.at(i).value()) {
      throw std::runtime_error("Unexpected RNS Modulus");
    }
  }

  std::size_t num_ciphertexts = 0;
  read_or_throw(reinterpret_cast<char *>(&num_ciphertexts),
                sizeof(num_ciphertexts));

  std::size_t ciphertexts_size_bytes = 0;
  read_or_throw(reinterpret_cast<char *>(&ciphertexts_size_bytes),
                sizeof(ciphertexts_size_bytes));
  auto memory_pool = memory_pool_create_callback(ciphertexts_size_bytes);

  for (std::size_t i = 0; i < num_ciphertexts; i++) {
    std::size_t name_size = 0;
    read_or_throw(reinterpret_cast<char *>(&name_size), sizeof(name_size));
    if (name_size == 0) {
      throw std::runtime_error("Invalid ciphertext name size");
    }

    std::vector<char> name_buf(name_size);
    read_or_throw(name_buf.data(), name_buf.size());
    if (name_buf.back() != '\0') {
      throw std::runtime_error("Invalid ciphertext name");
    }
    std::string name(name_buf.data());

    std::size_t num_limbs = 0;
    read_or_throw(reinterpret_cast<char *>(&num_limbs), sizeof(num_limbs));
    if (num_limbs > context_num_rns_bases) {
      throw std::runtime_error("Invalid ciphertext limb count: " + name);
    }

    std::size_t elt_count = 0;
    read_or_throw(reinterpret_cast<char *>(&elt_count), sizeof(elt_count));
    if (elt_count != context_n) {
      throw std::runtime_error("Unexpected Ciphertext Limb Size");
    }

    auto c0 = std::make_shared<RnsPolynomial>();
    auto c1 = std::make_shared<RnsPolynomial>();
    for (auto *component : {c0.get(), c1.get()}) {
      for (std::size_t j = 0; j < num_limbs; j++) {
        std::uint64_t limb_id = 0;
        read_or_throw(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
        if (limb_id >= context_num_rns_bases) {
          throw std::runtime_error("Invalid ciphertext RNS base id: " + name);
        }

        auto limb_data = Pointer<Limb::Element_t>(memory_pool, elt_count);
        read_or_throw(reinterpret_cast<char *>(limb_data.get()),
                      elt_count * sizeof(Limb::Element_t));
        auto limb = std::make_shared<Limb>(std::move(limb_data), elt_count,
                                            limb_id, true);
        component->write_limb(std::move(limb), limb_id);
      }
    }

    if (!ciphertexts.emplace(std::move(name), RnsPolynomialPair{c0, c1})
             .second) {
      throw std::runtime_error("Duplicate ciphertext name");
    }
  }

  return ciphertexts;
}

std::unordered_map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
IOGenerator::deserialize_evalkeys(
    const Context &context, const std::string &evalkeys_file_name,
    std::function<MemoryManagerHandle(size_t size)>
        memory_pool_create_callback) {

  std::unordered_map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
      evalkeys;
  std::ifstream evalkey_file(evalkeys_file_name,
                             std::ios::in | std::ios::binary);
  if (!evalkey_file.is_open()) {
    throw std::runtime_error(
        "Evalkeys file :" + evalkeys_file_name +
        " cannot be opened. Did you forget to generate the evalkeys?");
  }

  std::vector<char> evk_header_buf(evalkey_header.size() + 1);
  evalkey_file.read(evk_header_buf.data(), evalkey_header.size() + 1);
  std::string evk_header_read(evk_header_buf.data());
  if (evalkey_header != evk_header_read) {
    throw std::runtime_error("Unexpected Header");
  }

  std::size_t context_n = 0;
  evalkey_file.read(reinterpret_cast<char *>(&context_n), sizeof(context_n));
  if (context_n != context.n()) {
    throw std::runtime_error("Unexpected Context Size");
  }
  std::size_t context_num_rns_bases = 0;
  evalkey_file.read(reinterpret_cast<char *>(&context_num_rns_bases),
                    sizeof(context_num_rns_bases));
  if (context_num_rns_bases != context.num_rns_bases()) {
    throw std::runtime_error("Unexpected Number of RNS Bases");
  }
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < context_num_rns_bases; i++) {
    uint64_t modulus = 0;
    evalkey_file.read(reinterpret_cast<char *>(&modulus), sizeof(modulus));
    if (modulus != rns_bases.at(i).value()) {
      throw std::runtime_error("Unexpected RNS Modulus");
    }
  }

  std::size_t num_evalkeys = 0;
  evalkey_file.read(reinterpret_cast<char *>(&num_evalkeys),
                    sizeof(num_evalkeys));

  std::size_t evalkeys_size_bytes = 0;
  evalkey_file.read(reinterpret_cast<char *>(&evalkeys_size_bytes),
                    sizeof(evalkeys_size_bytes));

  auto memory_pool = memory_pool_create_callback(evalkeys_size_bytes);

  std::size_t bytes_read = 0;
  for (std::size_t i = 0; i < num_evalkeys; i++) {
    std::size_t evk_entry_id_size = 0;
    evalkey_file.read(reinterpret_cast<char *>(&evk_entry_id_size),
                      sizeof(evk_entry_id_size));

    std::vector<char> evk_entry_id_buf(evk_entry_id_size);
    evalkey_file.read(evk_entry_id_buf.data(), evk_entry_id_size);
    std::string evk_entry_id(evk_entry_id_buf.data());

    std::size_t num_limbs = 0;
    evalkey_file.read(reinterpret_cast<char *>(&num_limbs), sizeof(num_limbs));

    auto evk0 = std::make_shared<RnsPolynomial>();
    auto evk1 = std::make_shared<RnsPolynomial>();

    for (std::size_t j = 0; j < num_limbs; j++) {
      std::uint64_t limb_id = 0;
      evalkey_file.read(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
      // auto limb_data = allocate_limb_elts(context_n);
      auto limb_data = Pointer<Limb::Element_t>(memory_pool, context_n);
      evalkey_file.read(reinterpret_cast<char *>(limb_data.get()),
                        context_n * sizeof(Limb::Element_t));
      auto limb = std::make_shared<Limb>(std::move(limb_data), context_n,
                                         limb_id, true);
      evk0->write_limb(std::move(limb), limb_id);
    }

    for (std::size_t j = 0; j < num_limbs; j++) {
      std::uint64_t limb_id = 0;
      evalkey_file.read(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
      // auto limb_data = allocate_limb_elts(context_n);
      auto limb_data = Pointer<Limb::Element_t>(memory_pool, context_n);
      evalkey_file.read(reinterpret_cast<char *>(limb_data.get()),
                        context_n * sizeof(Limb::Element_t));
      auto limb = std::make_shared<Limb>(std::move(limb_data), context_n,
                                         limb_id, true);
      evk1->write_limb(std::move(limb), limb_id);
    }

    evalkeys[evk_entry_id] = std::pair(evk0, evk1);
  }
  evalkey_file.close();
  return evalkeys;
}

std::unordered_map<std::string, RnsPolynomialPtr>
IOGenerator::deserialize_plaintexts_no_copy(
    const Context &context, const std::string &plaintexts_file_name,
    std::function<MemoryManagerHandle(size_t size)>
        memory_pool_create_callback) {
  std::unordered_map<std::string, RnsPolynomialPtr> plaintexts;
  BufferedPlaintextsFile plaintexts_file(plaintexts_file_name);
  if (!plaintexts_file.is_open()) {
    throw std::runtime_error(
        "Plaintexts file: " + plaintexts_file_name +
        " cannot be opened. Did you forget to generate the plaintexts?");
  }

  std::vector<char> plaintext_header_buf(plaintext_header.size() + 1);
  plaintexts_file.read(plaintext_header_buf.data(),
                       plaintext_header.size() + 1);
  std::string plaintext_header_read(plaintext_header_buf.data());
  if (plaintext_header != plaintext_header_read) {
    throw std::runtime_error("Unexpected Header");
  }

  std::size_t context_n = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_n), sizeof(context_n));
  if (context_n != context.n()) {
    throw std::runtime_error("Unexpected Context Size");
  }
  std::size_t context_num_rns_bases = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_num_rns_bases),
                       sizeof(context_num_rns_bases));
  if (context_num_rns_bases != context.num_rns_bases()) {
    throw std::runtime_error("Unexpected Number of RNS Bases");
  }
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < context_num_rns_bases; i++) {
    uint64_t modulus = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&modulus), sizeof(modulus));
    if (modulus != rns_bases.at(i).value()) {
      throw std::runtime_error("Unexpected RNS Modulus");
    }
  }

  std::size_t num_plaintexts = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&num_plaintexts),
                       sizeof(num_plaintexts));
  num_plaintexts = 1;

  std::size_t plaintexts_size_bytes = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&plaintexts_size_bytes),
                       sizeof(plaintexts_size_bytes));

  auto memory_pool = memory_pool_create_callback(plaintexts_size_bytes);

  for (std::size_t i = 0; i < num_plaintexts; i++) {
    // auto evk_entry = evalkey_entries.at(k);
    std::size_t plaintext_name_size = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&plaintext_name_size),
                         sizeof(plaintext_name_size));

    std::vector<char> plaintext_name_buf(plaintext_name_size);
    plaintexts_file.read(plaintext_name_buf.data(), plaintext_name_size);
    std::string plaintext_name(plaintext_name_buf.data());

    std::size_t num_limbs = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&num_limbs),
                         sizeof(num_limbs));

    std::size_t elt_count = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&elt_count),
                         sizeof(elt_count));

    auto pt = std::make_shared<RnsPolynomial>();

    for (std::size_t j = 0; j < num_limbs; j++) {
      std::uint64_t limb_id = 0;
      plaintexts_file.read(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
      // auto limb_data = allocate_limb_elts(elt_count);
      auto limb_data = Pointer<Limb::Element_t>(memory_pool, elt_count);
      plaintexts_file.read(reinterpret_cast<char *>(limb_data.get()),
                           elt_count * sizeof(Limb::Element_t));
      // auto limb = std::make_shared<Limb>(std::move(limb_data), context_n, limb_id, true);
      auto limb = std::make_shared<Limb>(std::move(limb_data), elt_count,
                                         limb_id, true);
      pt->write_limb(std::move(limb), limb_id);
    }

    plaintexts[plaintext_name] = pt;
  }
  plaintexts_file.close();
  return plaintexts;
}

std::unordered_map<std::string, RnsPolynomialPtr>
IOGenerator::deserialize_plaintexts(
    const Context &context, const std::string &plaintexts_file_name,
    std::function<MemoryManagerHandle(size_t size)>
        memory_pool_create_callback) {

  std::unordered_map<std::string, RnsPolynomialPtr> plaintexts;
  BufferedPlaintextsFile plaintexts_file(plaintexts_file_name);
  if (!plaintexts_file.is_open()) {
    throw std::runtime_error(
        "Plaintexts file: " + plaintexts_file_name +
        " cannot be opened. Did you forget to generate the plaintexts?");
  }

  std::vector<char> plaintext_header_buf(plaintext_header.size() + 1);
  plaintexts_file.read(plaintext_header_buf.data(),
                       plaintext_header.size() + 1);
  std::string plaintext_header_read(plaintext_header_buf.data());
  if (plaintext_header != plaintext_header_read) {
    throw std::runtime_error("Unexpected Header");
  }

  std::size_t context_n = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_n), sizeof(context_n));
  if (context_n != context.n()) {
    throw std::runtime_error("Unexpected Context Size");
  }
  std::size_t context_num_rns_bases = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_num_rns_bases),
                       sizeof(context_num_rns_bases));
  if (context_num_rns_bases != context.num_rns_bases()) {
    throw std::runtime_error("Unexpected Number of RNS Bases");
  }
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < context_num_rns_bases; i++) {
    uint64_t modulus = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&modulus), sizeof(modulus));
    if (modulus != rns_bases.at(i).value()) {
      throw std::runtime_error("Unexpected RNS Modulus");
    }
  }

  std::size_t num_plaintexts = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&num_plaintexts),
                       sizeof(num_plaintexts));

  std::size_t plaintexts_size_bytes = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&plaintexts_size_bytes),
                       sizeof(plaintexts_size_bytes));

  auto memory_pool = memory_pool_create_callback(plaintexts_size_bytes);

  for (std::size_t i = 0; i < num_plaintexts; i++) {
    // auto evk_entry = evalkey_entries.at(k);
    std::size_t plaintext_name_size = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&plaintext_name_size),
                         sizeof(plaintext_name_size));

    std::vector<char> plaintext_name_buf(plaintext_name_size);
    plaintexts_file.read(plaintext_name_buf.data(), plaintext_name_size);
    std::string plaintext_name(plaintext_name_buf.data());

    std::size_t num_limbs = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&num_limbs),
                         sizeof(num_limbs));

    std::size_t elt_count = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&elt_count),
                         sizeof(elt_count));

    auto pt = std::make_shared<RnsPolynomial>();

    for (std::size_t j = 0; j < num_limbs; j++) {
      std::uint64_t limb_id = 0;
      plaintexts_file.read(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
      // auto limb_data = allocate_limb_elts(elt_count);
      auto limb_data = Pointer<Limb::Element_t>(memory_pool, elt_count);
      plaintexts_file.read(reinterpret_cast<char *>(limb_data.get()),
                           elt_count * sizeof(Limb::Element_t));
      // auto limb = std::make_shared<Limb>(std::move(limb_data), context_n, limb_id, true);
      auto limb = std::make_shared<Limb>(std::move(limb_data), elt_count,
                                         limb_id, true);
      pt->write_limb(std::move(limb), limb_id);
    }

    plaintexts[plaintext_name] = pt;
  }
  plaintexts_file.close();
  return plaintexts;
}
std::pair<std::unordered_map<std::string, RnsPolynomialPtr>, size_t>
IOGenerator::deserialize_plaintexts_old_format(
    const Context &context, const std::string &plaintexts_file_name) {

  std::unordered_map<std::string, RnsPolynomialPtr> plaintexts;
  BufferedPlaintextsFile plaintexts_file(plaintexts_file_name);
  if (!plaintexts_file.is_open()) {
    throw std::runtime_error(
        "Plaintexts file: " + plaintexts_file_name +
        " cannot be opened. Did you forget to generate the plaintexts?");
  }

  std::vector<char> plaintext_header_buf(plaintext_header.size() + 1);
  plaintexts_file.read(plaintext_header_buf.data(),
                       plaintext_header.size() + 1);
  std::string plaintext_header_read(plaintext_header_buf.data());
  if (plaintext_header != plaintext_header_read) {
    throw std::runtime_error("Unexpected Header");
  }

  std::size_t context_n = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_n), sizeof(context_n));
  if (context_n != context.n()) {
    throw std::runtime_error("Unexpected Context Size");
  }
  std::size_t context_num_rns_bases = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&context_num_rns_bases),
                       sizeof(context_num_rns_bases));
  if (context_num_rns_bases != context.num_rns_bases()) {
    throw std::runtime_error("Unexpected Number of RNS Bases");
  }
  const auto &rns_bases = context.rns_bases();
  for (int i = 0; i < context_num_rns_bases; i++) {
    uint64_t modulus = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&modulus), sizeof(modulus));
    if (modulus != rns_bases.at(i).value()) {
      throw std::runtime_error("Unexpected RNS Modulus");
    }
  }

  std::size_t num_plaintexts = 0;
  plaintexts_file.read(reinterpret_cast<char *>(&num_plaintexts),
                       sizeof(num_plaintexts));

  auto memory_pool = nullptr;
  size_t plaintexts_size = 0;
  for (std::size_t i = 0; i < num_plaintexts; i++) {
    // auto evk_entry = evalkey_entries.at(k);
    std::size_t plaintext_name_size = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&plaintext_name_size),
                         sizeof(plaintext_name_size));

    std::vector<char> plaintext_name_buf(plaintext_name_size);
    plaintexts_file.read(plaintext_name_buf.data(), plaintext_name_size);
    std::string plaintext_name(plaintext_name_buf.data());

    std::size_t num_limbs = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&num_limbs),
                         sizeof(num_limbs));

    std::size_t elt_count = 0;
    plaintexts_file.read(reinterpret_cast<char *>(&elt_count),
                         sizeof(elt_count));

    auto pt = std::make_shared<RnsPolynomial>();

    for (std::size_t j = 0; j < num_limbs; j++) {
      std::uint64_t limb_id = 0;
      plaintexts_file.read(reinterpret_cast<char *>(&limb_id), sizeof(limb_id));
      auto limb_data = Pointer<Limb::Element_t>(memory_pool, elt_count);
      plaintexts_file.read(reinterpret_cast<char *>(limb_data.get()),
                           elt_count * sizeof(Limb::Element_t));
      // auto limb = std::make_shared<Limb>(std::move(limb_data), context_n, limb_id, true);
      auto limb = std::make_shared<Limb>(std::move(limb_data), elt_count,
                                         limb_id, true);
      pt->write_limb(std::move(limb), limb_id);
      plaintexts_size += elt_count * sizeof(Limb::Element_t);
    }

    plaintexts[plaintext_name] = pt;
  }
  plaintexts_file.close();
  return {plaintexts, plaintexts_size};
}
} // namespace Runtime
} // namespace Cerium
