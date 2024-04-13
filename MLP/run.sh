NR_DPUS=2050lks  NR_TASKLETS=16 make -B
# ./bin/gemv_host -m 1000 -n 1000 -e 10 
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 64
./bin/gemv_host -m 10000 -n 10000 -e 10 -b 64
# NR_DPUS=2  NR_TASKLETS=2 make -B
# ./bin/gemv_host -m 4 -n 4 -e 1 
