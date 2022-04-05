#include <map>
#include <fstream>

#include "ooo_cpu.h"

constexpr std::size_t BIMODAL_TABLE_SIZE = 16384;
constexpr std::size_t BIMODAL_PRIME = 16381;
constexpr std::size_t COUNTER_BITS = 2;

std::map<O3_CPU*, std::array<int, BIMODAL_TABLE_SIZE>> bimodal_table;

const char PREDICT = 'P', LAST = 'L', INIT = 'I';
const string CSV_SEPERATOR = ", ";
const string TRACE_FILE_PATH = "./";
const string TRACE_FILE_NAME = "branch_calls";
const string TRACE_FILE_EXT = ".txt";

// struct Branch_details {
//     uint64_t ip, target;
//     uint8_t taken, type;
//     string API_type
// }

const bool USE_SEPERATE_FILES = true;

std::ofstream trace_file;
std::ofstream trace_files[2];


void O3_CPU::initialize_branch_predictor()
{
  std::cout << "CPU " << cpu << " Bimodal branch predictor" << std::endl;
  if (USE_SEPERATE_FILES) {
      trace_files[0].open(TRACE_FILE_PATH + TRACE_FILE_NAME + "_" + PREDICT + TRACE_FILE_EXT);
      trace_files[1].open(TRACE_FILE_PATH + TRACE_FILE_NAME + "_" + LAST + TRACE_FILE_EXT);
  } else {
      trace_file.open(TRACE_FILE_PATH + TRACE_FILE_NAME + TRACE_FILE_EXT);
  } 
  bimodal_table[this] = {};
}

void write_to_file(uint64_t ip, uint64_t target, uint8_t taken, uint8_t type, char API) {
    int file_stram_array_index;
    if (API == PREDICT) { file_stram_array_index = 0; }
    else { file_stram_array_index = 1; }
    if (USE_SEPERATE_FILES) {
        trace_files[file_stram_array_index] << API << CSV_SEPERATOR 
               << ip << CSV_SEPERATOR
               << target << CSV_SEPERATOR
               << (unsigned int)taken << CSV_SEPERATOR
               << (unsigned int)type << endl;
    } else {
        trace_file << API << CSV_SEPERATOR 
               << ip << CSV_SEPERATOR
               << target << CSV_SEPERATOR
               << (unsigned int)taken << CSV_SEPERATOR
               << (unsigned int)type << endl;
    }
    
}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  uint32_t hash = ip % BIMODAL_PRIME;
  write_to_file(ip, predicted_target, always_taken, branch_type, PREDICT);
  return bimodal_table[this][hash] >= (1 << (COUNTER_BITS - 1));
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  uint32_t hash = ip % BIMODAL_PRIME;
  write_to_file(ip, branch_target, taken, branch_type, LAST);

  if (taken)
    bimodal_table[this][hash] = std::min(bimodal_table[this][hash] + 1, ((1 << COUNTER_BITS) - 1));
  else
    bimodal_table[this][hash] = std::max(bimodal_table[this][hash] - 1, 0);
}
