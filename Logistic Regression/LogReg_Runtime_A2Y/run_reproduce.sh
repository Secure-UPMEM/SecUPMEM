#Generating the data in Fig 16
# m is the number of samples n is the number of features 
# l is the learning rate and i is the number of iterations
NR_DPUS=2048 NR_TASKLETS=16 make -B
./bin/host_code -m 16000 -n 16 -l 0.001 -i 1000
./bin/host_code -m 160000 -n 16 -l 0.001 -i 1000
NR_DPUS=2500 NR_TASKLETS=16 make -B
./bin/host_code -m 640000 -n 16 -l 0.001 -i 1000



