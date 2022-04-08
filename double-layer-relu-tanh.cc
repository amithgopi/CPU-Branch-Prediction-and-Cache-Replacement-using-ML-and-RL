#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <map>
#include <cmath>

#include "ooo_cpu.h"

template <typename T, std::size_t HISTLEN, std::size_t BITS>
class perceptron
{
  T bias = 0;
  std::array<T, HISTLEN> weights = {};

  T bias_h = 0;
  T weight_h, output_h;

  T output;

public:
  // maximum and minimum weight values
  constexpr static T max_weight = (1 << (BITS - 1)) - 1;
  constexpr static T min_weight = -(max_weight + 1);

  T predict(std::bitset<HISTLEN> history)
  {
    output_h = bias;
    output = bias_h;

    //Dot product of the history register and the perceptron weights.
    for (std::size_t i = 0; i < std::size(history); i++) {
      if (history[i]) output_h += weights[i];
      else output_h -= weights[i];
    }

    // "ReLU" activation
    output_h = std::max(0, output_h);

    output += weight_h*output_h;

    // "Tanh" activation
    output = std::tanh(output);

    return output;
  }

  void update(bool result, std::bitset<HISTLEN> history)
  {
    //auto pred = (output > 0) ? 1 : -1;
    auto pred = output;
    auto real = result ? 1 : -1;
    auto alpha = 0.001; // learning rate

    // Using hinge loss
    auto dLoss = (pred == real) ? 0 : -real;
    auto dOut = 1 - std::tanh(output_h*weight_h + bias_h)*std::tanh(output_h*weight_h + bias_h); // tanh
    auto dOut_h = (output_h > 0) ? 1 : 0; // ReLU

    // Update weights
    for (std::size_t i = 0; i < std::size(history); i++) {
      weights[i] -= alpha * dLoss * dOut * weight_h * dOut_h * history[i];
      weights[i] = std::min(weights[i], max_weight);
      weights[i] = std::max(weights[i], min_weight);
    }

    weight_h -= alpha * dLoss * dOut * output_h;
    weight_h = std::min(weight_h, max_weight);
    weight_h = std::max(weight_h, min_weight);

    // Update bias
    bias -= alpha * dLoss * dOut * weight_h * dOut_h * 1;
    bias = std::min(bias, max_weight);
    bias = std::max(bias, min_weight);

    bias_h -= alpha * dLoss * dOut * output_h;
    bias_h = std::min(bias_h, max_weight);
    bias_h = std::max(bias_h, min_weight);
  }

};

constexpr std::size_t PERCEPTRON_HISTORY = 24; // history length for the global history shift register
constexpr std::size_t PERCEPTRON_BITS = 8;     // number of bits per weight
constexpr std::size_t NUM_PERCEPTRONS = 163;   // total number of perceptrons

constexpr int THETA = 1.93 * PERCEPTRON_HISTORY + 14; // threshold for training

constexpr std::size_t NUM_UPDATE_ENTRIES = 100; // size of buffer for keeping 'perceptron_state' for update

/* 'perceptron_state' - stores the branch prediction and keeps information
 * such as output and history needed for updating the perceptron predictor
 */
struct perceptron_state {
  uint64_t ip = 0;
  bool prediction = false;                     // prediction: 1 for taken, 0 for not taken
  int output = 0;                              // perceptron output
  std::bitset<PERCEPTRON_HISTORY> history = 0; // value of the history register yielding this prediction
};

std::map<O3_CPU*, std::array<perceptron<int, PERCEPTRON_HISTORY, PERCEPTRON_BITS>, NUM_PERCEPTRONS>> perceptrons; // table of perceptrons
//std::map<03_CPU*, std::array<perceptrons, NUM_LAYERS>> layers;          // layers of perceptrons
std::map<O3_CPU*, std::deque<perceptron_state>> perceptron_state_buf;   // state for updating perceptron predictor
std::map<O3_CPU*, std::bitset<PERCEPTRON_HISTORY>> spec_global_history; // speculative global history - updated by predictor
std::map<O3_CPU*, std::bitset<PERCEPTRON_HISTORY>> global_history;      // real global history - updated when the predictor is
                                                                        // updated
void O3_CPU::initialize_branch_predictor() {}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  // hash the address to get an index into the table of perceptrons
  auto index = ip % NUM_PERCEPTRONS;
  auto output = perceptrons[this][index].predict(spec_global_history[this]);

  bool prediction = (output >= 0);

  // record the various values needed to update the predictor
  perceptron_state_buf[this].push_back({ip, prediction, output, spec_global_history[this]});
  if (std::size(perceptron_state_buf[this]) > NUM_UPDATE_ENTRIES)
    perceptron_state_buf[this].pop_front();

  // update the speculative global history register
  spec_global_history[this] <<= 1;
  spec_global_history[this].set(0, prediction);

  return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
  auto state = std::find_if(std::begin(perceptron_state_buf[this]), std::end(perceptron_state_buf[this]), [ip](auto x) { return x.ip == ip; });
  if (state == std::end(perceptron_state_buf[this])) return; // Skip update because state was lost

  auto [_ip, prediction, output, history] = *state;
  perceptron_state_buf[this].erase(state);

  auto index = ip % NUM_PERCEPTRONS;

  // update the real global history shift register
  global_history[this] <<= 1;
  global_history[this].set(0, taken);

  // if this branch was mispredicted, restore the speculative history to the last known real history
  if (prediction != taken) spec_global_history[this] = global_history[this];

  // adjust weights
  perceptrons[this][index].update(taken, history);
}
