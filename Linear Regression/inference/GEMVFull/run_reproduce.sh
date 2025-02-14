#Generating the data in Fig 17

echo " *********** Linear Regression inference **********"
NR_DPUS=1024 NR_TASKLETS=3 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V), Fig 13 and 14 (left)**********"
echo " "
echo " *********** Reproducing results for 20480 samples and 3 featureds **********"
./bin/gemv_host -m 20480 -n 3 -e 10
echo " "
echo " *********** Reproducing results for 40960 samples and 3 featureds **********"
./bin/gemv_host -m 40960 -n 3 -e 10