#Generating the data in Fig 17

NR_DPUS=1024 NR_TASKLETS=3 make -B
./bin/gemv_host -m 20480 -n 3 -e 10
./bin/gemv_host -m 40960 -n 3 -e 10