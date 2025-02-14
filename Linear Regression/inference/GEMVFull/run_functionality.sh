#this is just a simple test for functionality test

echo " *********** Linear Regression Inference **********"
NR_DPUS=1024 NR_TASKLETS=3 make -B
echo " "
echo " *********** Built  **********"
echo " "
echo " *********** Functionality test for 20480 samples and 3 features **********"
./bin/gemv_host -m 20480 -n 3 -e 10
