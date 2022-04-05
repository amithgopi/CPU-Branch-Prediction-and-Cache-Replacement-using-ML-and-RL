#include <algorithm>
#include <iterator>
#include <fstream>

#include "cache.h"
#include "util.h"

const string CSV_SEPERATOR = ", ";
const string TRACE_FILE_PATH = "./";
const string TRACE_FILE_NAME = "cache_calls";
const string TRACE_FILE_EXT = ".txt";
const char FIND_ = 'F', UPDATE_ = 'U', INIT_ = 'I';
// const bool USE_SEPERATE_FILES = true;

// std::ofstream trace_file;
std::ofstream trace_file_find, trace_file_update;

void CACHE::initialize_replacement() {

  trace_file_find.open(TRACE_FILE_PATH + TRACE_FILE_NAME + "_" + FIND_ + TRACE_FILE_EXT);
  trace_file_update.open(TRACE_FILE_PATH + TRACE_FILE_NAME + "_" + UPDATE_ + TRACE_FILE_EXT);

}

void write_to_file(uint32_t cpu, uint64_t instr_id, uint32_t set, uint32_t way, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint64_t victim_addr,
                                     uint32_t type, uint8_t hit, string origin, char API) {
    if (API == FIND_) {
      trace_file_find << cpu << CSV_SEPERATOR 
               << instr_id << CSV_SEPERATOR
               << set << CSV_SEPERATOR
               << current_set << CSV_SEPERATOR
               << ip << CSV_SEPERATOR
               << full_addr << CSV_SEPERATOR
               << type << CSV_SEPERATOR
               << origin << endl;
    } else if (API == UPDATE_) {
      trace_file_update << cpu << CSV_SEPERATOR 
               << set << CSV_SEPERATOR
               << way << CSV_SEPERATOR
               << full_addr << CSV_SEPERATOR
               << ip << CSV_SEPERATOR
               << full_addr << CSV_SEPERATOR
               << victim_addr << CSV_SEPERATOR
               << (unsigned int)hit << CSV_SEPERATOR
               << type << CSV_SEPERATOR
               << origin << endl;

    }
    
}

// find replacement victim
uint32_t CACHE::find_victim(uint32_t cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type, string origin)
{
  write_to_file(cpu, instr_id, set, 0, current_set, ip, full_addr, 0, type, 0, origin, FIND_);
  // baseline LRU
  return std::distance(current_set, std::max_element(current_set, std::next(current_set, NUM_WAY), lru_comparator<BLOCK, BLOCK>()));
}

// called on every cache hit and cache fill
void CACHE::update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit, string origin)
{
  write_to_file(cpu, 0, set, way, nullptr, ip, full_addr, victim_addr, type, hit, origin, UPDATE_);
  if (hit && type == WRITEBACK)
    return;

  auto begin = std::next(block.begin(), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  uint32_t hit_lru = std::next(begin, way)->lru;
  std::for_each(begin, end, [hit_lru](BLOCK& x) {
    if (x.lru <= hit_lru)
      x.lru++;
  });
  std::next(begin, way)->lru = 0; // promote to the MRU position
}

void CACHE::replacement_final_stats() {

}