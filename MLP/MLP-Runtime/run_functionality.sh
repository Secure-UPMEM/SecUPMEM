##this is just a simple functionality test. 

echo " *********** MLP-Runtime **********"
NR_DPUS=1024  NR_TASKLETS=16 make -B
echo " "
echo " *********** Built  **********"
echo " "
echo " *********** Functionality test for matrix of 1024 * 1024 **********"
./bin/gemv_host -m 1024 -n 1024 -e 10 -b 32