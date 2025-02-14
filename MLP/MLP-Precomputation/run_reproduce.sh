#Generating the data in Fig 13 and 14 (Left)
# m is the number of rows n is the number of columns 
# e is the number of iterations and b is the batch size

echo " *********** MLP-Precomputation **********"
echo " "

NR_DPUS=2050  NR_TASKLETS=16 make -B
echo " *********** Built, Generating UPMEM-Precompute-C(V), Fig 13 and 14 (left)**********"
echo " "
echo " *********** Reproducing results for model size 4 KB/ 4 MB and batch size = 64 **********"
./bin/gemv_host -m 1000 -n 1000 -e 10 -b 64
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 64 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 64
echo " "
echo " *********** Reproducing results for model size 40 KB/ 400 MB and batch size = 64 **********"
./bin/gemv_host -m 10000 -n 10000 -e 10 -b 64


# Generating data in Fig 14 (Right)
NR_DPUS=2150  NR_TASKLETS=16 make -B
echo " *********** Built, Generating UPMEM-Precompute-C(V), Fig 14 (right) **********"
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 32 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 32
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 64 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 64
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 128 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 128
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 256 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 256
echo " "
echo " *********** Reproducing results for model size 20 KB/ 100 MB and batch size = 512 **********"
./bin/gemv_host -m 5000 -n 5000 -e 10 -b 512