##this is just a simple test for functionality test

NR_DPUS=1024  NR_TASKLETS=16 make -B
./bin/gemv_host -m 1024 -n 1024 -e 10 -b 32