#this is just a simple test for functionality test
NR_DPUS=1024 NR_TASKLETS=16 make -B
./bin/host_code -m 10240 -n 16 -l 0.001 -i 100