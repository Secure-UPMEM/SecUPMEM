#this is just a simple test for functionality test
sudo apt install  -y libgmp-dev
NR_DPUS=1024 NR_TASKLETS=16 make -B
./bin/host_code -m 4096 -n 16 -l 0.001 -i 100