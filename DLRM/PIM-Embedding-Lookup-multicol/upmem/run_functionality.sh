#this is just a simple test for functionality test do not expect acceleration
echo " *********** DLRM-Both **********"
make clean 
INDICES_PER_LOOKUP=32 EMBEDDING_DEPTH=1000000 NR_RUN=1 BATCH_SIZE=128 NR_EMBEDDING=64 make -B
echo " "
echo " *********** Built  **********"
echo " "
echo " *********** Functionality test for Embedding tables of 64*1000000 and BATCH_SIZE=128 and numb_indices = 32  **********"
make testdpu