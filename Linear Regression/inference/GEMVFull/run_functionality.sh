#this is just a simple test for functionality test

NR_DPUS=1024 NR_TASKLETS=3 make -B
./bin/host_code -m 20480 -n 3 -l 0.001 -i 100