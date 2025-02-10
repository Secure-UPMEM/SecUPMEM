#this is just a simple test for functionality test do not expect acceleration
make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=1000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
make testdpu