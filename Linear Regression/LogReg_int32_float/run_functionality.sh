#this is just a simple test for functionality test
sudo apt install  -y libgmp-dev

echo " *********** Linear Regression training **********"
NR_DPUS=1024 NR_TASKLETS=16 make -B
echo " *********** Built **********"
echo " "
echo " *********** Functionality test results for 4096 samples and 16 featureds **********"
./bin/host_code -m 4096 -n 16 -l 0.001 -i 100