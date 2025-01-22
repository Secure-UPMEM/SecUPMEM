NR_DPUS=2432  NR_TASKLETS=16 make -B
./bin/gemv_host -m 1000 -n 1000 -e 10 -b 64
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 64
# ./bin/gemv_host -m 10000 -n 10000 -e 10 -b 64
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 32
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 128
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 256
# ./bin/gemv_host -m 5000 -n 5000 -e 10 -b 512

# NR_DPUS=2  NR_TASKLETS=2 make -B
# ./bin/gemv_host -m 4 -n 4 -e 1 
