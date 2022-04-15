#include "ooo_cpu.h"

#define ALPHA 0.2
#define Q_TABLE_SIZE 16384
float qtable[Q_TABLE_SIZE][2] = {0};

#define GLOBAL_HISTORY_LENGTH 14
#define GLOBAL_HISTORY_MASK (1 << GLOBAL_HISTORY_LENGTH) - 1
#define GLOBAL_HISTORY_TABLE_SIZE 16384
int global_history_table[GLOBAL_HISTORY_TABLE_SIZE] = {0};

int branch_history = 0;

void O3_CPU::initialize_branch_predictor()
{
    cout << "CPU " << cpu << " G-QLAg branch predictor" << endl;
}

// TODO : Update hash logic
unsigned int get_hash(uint64_t ip, int branch_history)
{
    unsigned int hash;

    hash = ip ^ (ip >> GLOBAL_HISTORY_LENGTH) ^ (ip >> (GLOBAL_HISTORY_LENGTH * 2)) ^ branch_history;
    hash = hash % GLOBAL_HISTORY_TABLE_SIZE;

    return hash;
}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
    bool prediction;

    unsigned int index = get_hash(ip, branch_history);
    //prediction = (global_history_table[index] > 1) ? 1 : 0;
    prediction = (qtable[index][0] > qtable[index][1]) ? 0 : 1;

    return prediction;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{
    unsigned int index = get_hash(ip, branch_history);

    if (taken)
    {
        if (qtable[index][1] > qtable[index][0])
            qtable[index][1] = (1 - ALPHA)*qtable[index][1] + ALPHA;
        else
            qtable[index][0] = (1 - ALPHA)*qtable[index][0] - ALPHA;
    }

    else
    {
        if (qtable[index][0] > qtable[index][1])
            qtable[index][0] = (1 - ALPHA)*qtable[index][0] + ALPHA;
        else
            qtable[index][1] = (1 - ALPHA)*qtable[index][1] - ALPHA;
    }

    // TODO : set the range of q values

    // update branch history vector
    branch_history <<= 1;
    branch_history &= GLOBAL_HISTORY_MASK;
    branch_history |= taken;
}
