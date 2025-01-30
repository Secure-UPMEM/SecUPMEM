NR_DPUS=2048 NR_TASKLETS=16 make -B
./bin/host_code -m 16000 -n 16 -l 0.001 -i 1000
./bin/host_code -m 160000 -n 16 -l 0.001 -i 1000
./bin/host_code -m 640000 -n 16 -l 0.001 -i 1000

#  NR_DPUS=2  NR_TASKLETS=2 make -B
# ./bin/host_code -m 4 -n 2 -l 0.001 -i 2


