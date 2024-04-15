#!/bin/bash

# Run the programs in the background and measure their execution times
( time ./run.sh -brdd random > program1_output.txt) 2> program1_execution_time.txt &
( time ./run.sh -brdd random > program2_output.txt) 2> program2_execution_time.txt &
( time ./run.sh -brdd random > program3_output.txt) 2> program3_execution_time.txt &
( time ./run.sh -brdd random > program4_output.txt) 2> program4_execution_time.txt &

# Wait for all background jobs to finish
wait

# Display a message indicating that all programs have finished
echo "All programs have finished execution."