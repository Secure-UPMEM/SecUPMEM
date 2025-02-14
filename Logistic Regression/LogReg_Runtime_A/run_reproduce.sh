#Generating the data in Fig 16
# m is the number of samples n is the number of features 
# l is the learning rate and i is the number of iterations
echo " *********** Logistic Regression training (A) **********"

NR_DPUS=2048 NR_TASKLETS=16 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V), Fig 16 **********"
echo " "
echo " *********** Reproducing results for 16000 samples and 16 featureds **********"
./bin/host_code -m 16000 -n 16 -l 0.001 -i 1000
echo " "
echo " *********** Reproducing results for 160000 samples and 16 featureds **********"
./bin/host_code -m 160000 -n 16 -l 0.001 -i 1000
echo " "
echo " *********** Reproducing results for 640000 samples and 16 featureds **********"
./bin/host_code -m 640000 -n 16 -l 0.001 -i 1000



