NR_DPUS=2048 NR_TASKLETS=16 make -B
./bin/host_code -m 40960 -n 16 -l 0.001 -i 1000
# ./bin/host_code -m 409600 -n 16 -l 0.001 -i 1000
# NR_DPUS=2432 NR_TASKLETS=16 make -B
# ./bin/host_code -m 819200 -n 16 -l 0.001 -i 1000


# NR_DPUS=4 NR_TASKLETS=2 make -B
# ./bin/host_code -m 8 -n 2 -l 0.001 -i 1000
