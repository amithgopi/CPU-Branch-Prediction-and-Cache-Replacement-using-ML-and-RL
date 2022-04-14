
#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>
#include <cmath>

#include "ooo_cpu.h"
#include "cache.h"
#include "util.h"

enum Perceptron_Features { PC_0, PC_1, PC_2, PC_3, TAG_4, TAG_7 };

// Perceptron parameters
constexpr std::size_t PC_HISTORY_SIZE = 4;
constexpr std::size_t PERCEPTRON_BITS = 6;     // number of bits per weight
constexpr std::size_t NUM_TABLE_ENTRIES = 256;
constexpr std::size_t NUM_TABLE_INDEX_BITS = 8;
constexpr std::size_t NUM_FEATURES = 6;
// These are updated in the init call
std::size_t LLC_NUM_SETS = 2048;
std::size_t LLC_OFFSET_BITS = 6;
std::size_t LLC_INDEX_BITS = 11;
// Thresholds for traning and prediction
constexpr int THETA = 74; // threshold for training
constexpr int TAU_REPLACE = 124;
constexpr int TAU_BYPASS = 3;

// This holds the input feature set for the percptron
struct input_features {
  uint64_t ip_0, ip_1, ip_2, ip_3;
  uint64_t tag, tag_shit_4, tag_shift_7;
};

//Queue to hold PC history
std::deque<uint64_t> PC_history;

void push_to_pc_histoy(uint64_t ip) {
  PC_history.push_front(ip);
  if (PC_history.size() > PC_HISTORY_SIZE) PC_history.pop_back();
}

class Perceptron
{
  int bias = 0;
  std::array<array<int, NUM_TABLE_ENTRIES>, NUM_FEATURES> weight_tables = {};

public:
  // maximum and minimum weight values
  constexpr static int max_weight = (1 << (PERCEPTRON_BITS - 1)) - 1;
  constexpr static int min_weight = -(max_weight + 1);
  const unsigned int bit_mask = ((1 << NUM_TABLE_INDEX_BITS) - 1);

  input_features extract_features(uint64_t ip, uint64_t address, bool is_find) {
    // TODO check if PC updated
    input_features features;
    if (is_find) {
      features.ip_0 = ((ip >> 2) ^ ip) & bit_mask;
      features.ip_1 = ((PC_history[0] >> 1) ^ ip) & bit_mask;
      features.ip_2 = ((PC_history[1] >> 2) ^ ip) & bit_mask;
      features.ip_3 = ((PC_history[2] >> 3) ^ ip) & bit_mask;
    } else {
      features.ip_0 = ((PC_history[0] >> 2) ^ ip) & bit_mask;
      features.ip_1 = ((PC_history[1] >> 1) ^ ip) & bit_mask;
      features.ip_2 = ((PC_history[2] >> 2) ^ ip) & bit_mask;
      features.ip_3 = ((PC_history[3] >> 3) ^ ip) & bit_mask;
    }
    int index_bits = log2(LLC_NUM_SETS);
    uint64_t tag = address >> (index_bits + LLC_OFFSET_BITS);
    features.tag = tag;
    features.tag_shit_4 = ((tag >> 4) ^ ip) & bit_mask;
    features.tag_shift_7 = ((tag >> 7) ^ ip) & bit_mask;
    return features;
  }

  int predict(input_features features)
  {
    int output = 0;  
    
    output += weight_tables[0][features.ip_0];
    output += weight_tables[1][features.ip_1];
    output += weight_tables[2][features.ip_2];
    output += weight_tables[3][features.ip_3];
    output += weight_tables[4][features.tag_shit_4];
    output += weight_tables[5][features.tag_shift_7];
    return output;
  }

  void train_decrement_weights(input_features features) {
    weight_tables[0][features.ip_0] = max(weight_tables[0][features.ip_0]-1, min_weight);
    weight_tables[1][features.ip_1] = max(weight_tables[0][features.ip_1]-1, min_weight);
    weight_tables[2][features.ip_2] = max(weight_tables[0][features.ip_2]-1, min_weight);
    weight_tables[3][features.ip_3] = max(weight_tables[0][features.ip_3]-1, min_weight);
    weight_tables[4][features.tag_shit_4] = max(weight_tables[0][features.tag_shit_4]-1, min_weight);
    weight_tables[5][features.tag_shift_7] = max(weight_tables[0][features.tag_shift_7]-1, min_weight);
  }

  void train_increment_weights(input_features features) {
    weight_tables[0][features.ip_0] = min(weight_tables[0][features.ip_0]+1, max_weight);
    weight_tables[1][features.ip_1] = min(weight_tables[0][features.ip_1]+1, max_weight);
    weight_tables[2][features.ip_2] = min(weight_tables[0][features.ip_2]+1, max_weight);
    weight_tables[3][features.ip_3] = min(weight_tables[0][features.ip_3]+1, max_weight);
    weight_tables[4][features.tag_shit_4] = min(weight_tables[0][features.tag_shit_4]+1, max_weight);
    weight_tables[5][features.tag_shift_7] = min(weight_tables[0][features.tag_shift_7]+1, max_weight);
  }

};

Perceptron perceptron; 

const std::size_t SAMPLER_SETS = 64;
const std::size_t SAMPLER_ASSOCIATIVITY = 16;
const std::size_t SAMPLER_ASSOCIATIVITY_BITS = log2(SAMPLER_ASSOCIATIVITY);

class Sampler {

public:
  const std::size_t SAMPLER_SET_INTERVAL = LLC_NUM_SETS/SAMPLER_SETS;
  const std::uint64_t PARTIAL_TAG_MASK = (1 << 15) - 1;


  
  struct sampler_entry {
    uint16_t tag;
    uint16_t y_out;
    input_features features;
    uint8_t lru = SAMPLER_ASSOCIATIVITY-1;
    bool valid = false;
  };

  std::array<array<sampler_entry, SAMPLER_ASSOCIATIVITY>, SAMPLER_SETS> sets;

  inline bool isSamplerSet(unsigned int set_index) {
    return set_index % SAMPLER_SET_INTERVAL == 0;
  }

  int find_in_sampler(uint16_t partial_tag, unsigned int index) {
    for (int i =0; i<SAMPLER_ASSOCIATIVITY; i++) {
      if(partial_tag == this->sets[index][i].tag && this->sets[index][i].valid) {
        return i;
      }
    }
    return -1;
  }

  inline int find_invalid_block(unsigned int set) {
    for (int i=0; i<SAMPLER_ASSOCIATIVITY; i++) {
      if (this->sets[set][i].valid == false) {
        return i;
      }
    }
    return -1;
  }

  inline int find_dead_block(unsigned int set) { 
    for (int i=0; i<SAMPLER_ASSOCIATIVITY; i++) {
      if (this->sets[set][i].y_out > TAU_REPLACE) {
        return i;
      }
    }
    return -1;
  }

  void update_lru(unsigned int set, unsigned int way) {
    unsigned int position = this->sets[set][way].lru;
    for (int i=0; i<SAMPLER_ASSOCIATIVITY; i++) {
      if (this->sets[set][way].lru < position) {
        this->sets[set][way].lru++;
      }
    }
    this->sets[set][way].lru = 0;
  }

  unsigned int find_lru_block(unsigned int set) {
    for (int way=0; way<SAMPLER_ASSOCIATIVITY; way++) {
      if (this->sets[set][way].lru == SAMPLER_ASSOCIATIVITY-1) {
        return way;
      }
    }
    return -1;
  }
};

Sampler sampler;

void CACHE::update_cache_lru(uint32_t set, uint32_t way) {
  auto begin = std::next(block.begin(), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  uint32_t hit_lru = std::next(begin, way)->lru;
  std::for_each(begin, end, [hit_lru](BLOCK& x) {
    if (x.lru <= hit_lru)
      x.lru++;
  });
  std::next(begin, way)->lru = 0; // promote to the MRU position
}

void CACHE::initialize_replacement() {
  LLC_NUM_SETS = NUM_SET;
  LLC_OFFSET_BITS = OFFSET_BITS;
  LLC_INDEX_BITS = (int)log2(LLC_NUM_SETS);
}

const bool ENABLE_BYPASS = false;
// find replacement victim
uint32_t CACHE::find_victim(uint32_t cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
  int way = 0;
  input_features features = perceptron.extract_features(ip, full_addr, false);
  int y_out = perceptron.predict(features);
  // cout<<"** Find Victim "<<instr_id<<", ip: "<<ip<<", addr: "<<full_addr<<" y_out: "<<y_out;
  if (y_out > TAU_BYPASS && ENABLE_BYPASS) {
    // cout<<" bypass";
    push_to_pc_histoy(ip);
    way = NUM_WAY;
  } else {
    // cout<<" no_bypass "<<set<<" "<<set*NUM_WAY<<" ";
    bool dead_block_found = false;
    for (int i=0; i<NUM_WAY; i++) {
      // cout<<(block[set*NUM_WAY + i].reuse ? 1 : 0)<<" ";
      if (block[set*NUM_WAY + i].reuse == false) {
        way = i;
        dead_block_found = true;
        break;
      }
    }
    if (dead_block_found == false) {
      // cout<<" LRU";
      way = std::distance(current_set, std::max_element(current_set, std::next(current_set, NUM_WAY), lru_comparator<BLOCK, BLOCK>()));
    }
  }
  // cout<<endl;
  return way;
}

// called on every cache hit and cache fill
void CACHE::update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{

  push_to_pc_histoy(ip);
  input_features features = perceptron.extract_features(ip, full_addr, true);
  int y_out = perceptron.predict(features);

  if (sampler.isSamplerSet(set)) {
    unsigned int sampler_set_index = set / sampler.SAMPLER_SET_INTERVAL;
    uint16_t partial_tag = full_addr & sampler.PARTIAL_TAG_MASK;
    
    int sampler_way = sampler.find_in_sampler(partial_tag, sampler_set_index);
    bool is_block_in_sampler = sampler_way != -1;
    if(is_block_in_sampler) {
      if (sampler.sets[sampler_set_index][sampler_way].y_out > (-1)*THETA) {
      // if (sampler.sets[sampler_set_index][sampler_way].y_out > -THETA || block[set*NUM_WAY + way].reuse != hit) {  //Enable prediction result check with threshold
        perceptron.train_decrement_weights(sampler.sets[sampler_set_index][sampler_way].features);
      }
      // sampler.sets[sampler_set_index][sampler_way].features = features;
      // sampler.update_lru(sampler_set_index, sampler_way);
      sampler.sets[sampler_set_index][sampler_way].y_out = y_out;
    } else {
      sampler_way = -1;
      sampler_way = sampler.find_invalid_block(sampler_set_index);
      if (sampler_way == -1) sampler_way = sampler.find_dead_block(sampler_set_index);
      if (sampler_way == -1) sampler_way = sampler.find_lru_block(sampler_set_index);
      if (sampler.sets[sampler_set_index][sampler_way].y_out < THETA || block[set*NUM_WAY + way].reuse != hit) {
        perceptron.train_increment_weights(sampler.sets[sampler_set_index][sampler_way].features);
      }
      sampler.sets[sampler_set_index][sampler_way].tag = partial_tag;
      sampler.sets[sampler_set_index][sampler_way].valid = true;
      
    }
    sampler.update_lru(sampler_set_index, sampler_way);
    sampler.sets[sampler_set_index][sampler_way].features = features;
    sampler.sets[sampler_set_index][sampler_way].y_out = y_out;
    
  }
  update_cache_lru(set, way);

  if (y_out < TAU_REPLACE) {
    block[set*NUM_WAY + way].reuse = true;
  } else {
    block[set*NUM_WAY + way].reuse = false;
  }
  // cout<<"Set: "<<set<<" block: "<<set*NUM_WAY + way<<" y_out: "<<y_out<<" reuse: "<<block[set*NUM_WAY + way].reuse<<endl;

}

void CACHE::replacement_final_stats() {} 
