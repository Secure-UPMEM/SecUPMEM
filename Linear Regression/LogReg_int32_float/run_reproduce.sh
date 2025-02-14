#Generating the data in Fig 18
# m is the number of samples n is the number of features 
# l is the learning rate and i is the number of iterations

echo " *********** Linear Regression training **********"

sudo apt install  -y libgmp-dev
NR_DPUS=2048 NR_TASKLETS=16 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V), Fig 18 **********"
echo " "
echo " *********** Reproducing results for 40960 samples and 16 featureds **********"
./bin/host_code -m 40960 -n 16 -l 0.001 -i 1000
echo " *********** Built, Generating UPMEM-Runtime-C(V), Fig 18 **********"
echo " "
echo " *********** Reproducing results for 409600 samples and 16 featureds **********"
./bin/host_code -m 409600 -n 16 -l 0.001 -i 1000
NR_DPUS=2432 NR_TASKLETS=16 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V), Fig 18 **********"
echo " "
echo " *********** Reproducing results for 819200 samples and 16 featureds **********"
./bin/host_code -m 819200 -n 16 -l 0.001 -i 1000