#!/bin/bash
#SBATCH --job-name=branch_rl_training   # Job name
#SBATCH --mail-type=END,FAIL         # Mail Events (NONE,BEGIN,FAIL,END,ALL)
#SBATCH --mail-user=abbypjoby@tamu.edu   # Replace with your email address
#SBATCH --ntasks=1                   # Run on a single CPU
#SBATCH --time=10:00:00              # Time limit hh:mm:ss
#SBATCH --output=serial_test_%j.log  # Standard output and error log
#SBATCH --partition=non-gpu          # This job does not use a GPU

echo "Running 600.perlbench_s-210B.champsimtrace.xz"
bin/champsim \
  -warmup_instructions 200000 \
  -simulation_instructions 10000000 \
  -traces ~pgratz/dpc3_traces/600.perlbench_s-210B.champsimtrace.xz \
  > 600.perlbench_s-210B.txt

### Add more ChampSim runs below.

