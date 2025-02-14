# Lines 3-21 are used to generate our results for UPMEM-Precompute-(CV) and UPMEM-Runtime-C(V) in figure 15(a)
echo " *********** DLRM-Both **********"

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=3000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (left)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*3000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=4000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (left)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*4000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=5000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (left)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*5000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (left)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=7000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (left)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*7000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu

# Lines 25-47 are used to generate our results for UPMEM-Precompute-(CV) and UPMEM-Runtime-C(V) in figure 15(b)

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=16 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=16 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=32 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=32 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=64 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=64 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=90 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=90 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=100 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=100 and numb_indices = 32 **********"
make testdpu

make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=6000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " *********** Built, Generating UPMEM-Runtime-C(V) and UPMEM-Precompute-C(V), Fig 15 (right)**********"
echo " "
echo " *********** Reproducing results for Embedding tables of 64*6000000 and BATCH_SIZE=128 and numb_indices = 32 **********"
make testdpu
