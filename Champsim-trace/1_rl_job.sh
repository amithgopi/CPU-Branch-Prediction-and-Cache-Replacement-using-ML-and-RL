#!/bin/bash
#SBATCH --job-name=serial_job_test   # Job name
#SBATCH --mail-type=END,FAIL         # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=abbypjoby@tamu.edu   # Replace with your email address
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --time=12:00:00              # Time limit hh:mm:ss
#SBATCH --output=serial_test_%j.log  # Standard output and error log
#SBATCH --partition=non-gpu          # This job does not use a GPU


echo "Running 619.lbm_s-4268B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 2000000 \
  -traces ~pgratz/dpc3_traces/619.lbm_s-4268B.champsimtrace.xz \
  > rl_619.txt


echo "Running 620.omnetpp_s-874B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 2000000 \
  -traces ~pgratz/dpc3_traces/620.omnetpp_s-874B.champsimtrace.xz \
  > rl_620.txt


echo "Running 621.wrf_s-575B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 2000000 \
  -traces ~pgratz/dpc3_traces/621.wrf_s-575B.champsimtrace.xz \
  > rl_621.txt


echo "Running 623.xalancbmk_s-700B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 2000000 \
  -traces ~pgratz/dpc3_traces/623.xalancbmk_s-700B.champsimtrace.xz \
  > rl_623.txt


echo "Running 625.x264_s-18B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 2000000 \
  -traces ~pgratz/dpc3_traces/625.x264_s-18B.champsimtrace.xz \
  > rl_625.txt