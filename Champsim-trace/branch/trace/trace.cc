#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cstring>
#include <vector>

#include "ooo_cpu.h"

#include "pipes_data.h"

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




///////


/* 
APIs to be accessed by the script.
*/

static void foo_api(std::ostream &os, const std::string &arg)
{
    os << "Foo was called with arg " << arg;
}

/* end API section */


/* return true if val is set, false for EOF */
static bool read_uint32(int read_fd, uint32_t &val)
{
    unsigned char msgSizeBuf[4];
    unsigned iBuf = 0;

    while (iBuf < sizeof(msgSizeBuf))
    {
        ssize_t rc = ::read(read_fd, msgSizeBuf + iBuf, sizeof(msgSizeBuf) - iBuf);

        if (rc == 0)
        {
            return false;
        }
        else if (rc < 0 )
        {
            std::cout << __func__ << "@" << __LINE__ << ":::Read ERROR" << std::endl;
            exit(1);
        }
        else
        {
            iBuf += rc;
        }
    }

    val = *(static_cast<uint32_t *>(static_cast<void *>(&msgSizeBuf[0])));
    
    return true;
}


static void send_msg(int write_fd, std::string msg)
{
    uint32_t msgSize = msg.size();
    unsigned char msgSizeBuf[4];

    ::memcpy(msgSizeBuf, &msgSize, sizeof(msgSize));

    unsigned iBuf = 0;
    while (iBuf < 4)
    {
        ssize_t rc = ::write(write_fd, msgSizeBuf + iBuf, sizeof(msgSizeBuf) - iBuf);
        if ( rc < 0 )
        {
            std::cout << "Error writing message size" << std::endl;
            ::exit(1);
        }
        else if ( rc == 0 )
        {
            std::cout << "rc == 0, what does that mean?" << std::endl;
            ::exit(1);
        }
        else
        {
            iBuf += rc;
        }
    }

    iBuf = 0;
    const char *msgBuf = msg.c_str();
    while (iBuf < msgSize)
    {
        ssize_t rc = ::write(write_fd, msgBuf + iBuf, msgSize - iBuf);
        if ( rc < 0 )
        {
            std::cout << "Error writing message" << std::endl;
            ::exit(1);
        }
        else if ( rc == 0 )
        {
            std::cout << "rc == 0, what does that mean?" << std::endl;
            ::exit(1);
        }
        else
        {
            iBuf += rc;
        }
    }
}

static std::string read_string(int read_fd, uint32_t sz)
{
    std::vector<char> msgBuf( sz + 1 );
    msgBuf[ sz ] = '\0';
    unsigned iBuf = 0;

    while (iBuf < sz)
    {
        ssize_t rc = ::read(read_fd, &(msgBuf[0]) + iBuf, sz - iBuf);

        if ( rc == 0 )
        {
            std::cout << __func__ << "@" << __LINE__ << ":::EOF read" << std::endl;
            exit(1);
        }
        else if ( rc < 0 )
        {
            std::cout << __func__ << "@" << __LINE__ << ":::Read ERROR during message" << std::endl;
            exit(1);
        }
        else
        {
            iBuf += rc;
        }
    }

    return std::string( &(msgBuf[0]) );
}


///////


uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type)
{
  uint32_t hash = ip % BIMODAL_PRIME;
  write_to_file(ip, predicted_target, always_taken, branch_type, PREDICT);

  //////////////////////////////////////////////////////////////////////////////
  //                    Send data to python agent through pipe
  //////////////////////////////////////////////////////////////////////////////
  //action here - send data to Python code
  std::ostringstream os;
  os << "test " << ip;
  std::cout<<"C++ Sending to pipe " << write_pipe << "\n";
  send_msg(write_pipe, os.str());
  //////////////////////////////////////////////////////////////////////////////

  return bimodal_table[this][hash] >= (1 << (COUNTER_BITS - 1));
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type)
{



  //////////////////////////////////////////////////////////////////////////////
  //                    Read data from python agent through pipe
  //////////////////////////////////////////////////////////////////////////////
  //reward here
  uint32_t apiNameSize;
  if (!read_uint32(read_pipe, apiNameSize))
  {
      // EOF waiting for a message, script ended
      std::cout << "EOF waiting for message, script ended" << std::endl;
      return;
  }
  std::string apiName = read_string(read_pipe, apiNameSize);
  uint32_t apiArgSize;
  if (!read_uint32(read_pipe, apiArgSize))
  {
      std::cout << "EOF white reading apiArgSize" << std::endl;
      ::exit(1);
  }
  std::string apiArg = read_string(read_pipe, apiArgSize);

  // std::cout<<"\nC++ read from pipe " << apiName << apiArg;


  // // Response comes as [resultSize][resultString]
  if (apiName == "predict_branch")
  {
      std::cout << "C++ Read from pipe :" << apiArg << std::endl;
  }
  else
  {
      std::cout << "UNSUPPORTED API " << apiName << std::endl;
  }

//////////////////////////////////////////////////////////////////////////////


  uint32_t hash = ip % BIMODAL_PRIME;
  write_to_file(ip, branch_target, taken, branch_type, LAST);

  if (taken)
    bimodal_table[this][hash] = std::min(bimodal_table[this][hash] + 1, ((1 << COUNTER_BITS) - 1));
  else
    bimodal_table[this][hash] = std::max(bimodal_table[this][hash] - 1, 0);


}
