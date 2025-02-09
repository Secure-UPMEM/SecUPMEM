### 

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDICES_PER_LOOKUP=160
MAX_BATCH_SIZE=60
EMBEDDING_DIM=64
BATCH_SIZE=8
NR_RUN=5
MAX_INDICES_PER_LOOKUP=120
MAX_INDICES_PER_LOOKUP_RAND=160
EMBEDDING_DEPTH=500000
NR_EMBEDDING=32
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 MAX_DPU_EMB_TABLE_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 96
nr cols per dpu 28
alloc dpus 96
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference

max nr embedding 2000
dpu [ms]: 2.089493, cpu [ms] 2.647592, dpu acceleration 1.267098
 DPU PRATIO 145.502282, CPU PRATIO 270.004226, DPU OK ? 1 
dpu [ms]: 2.086909, cpu [ms] 2.505759, dpu acceleration 1.200703
 DPU PRATIO 140.379461, CPU PRATIO 272.473461, DPU OK ? 1 
dpu [ms]: 2.094989, cpu [ms] 2.559729, dpu acceleration 1.221834
 DPU PRATIO 140.358605, CPU PRATIO 274.948369, DPU OK ? 1 
dpu [ms]: 2.085617, cpu [ms] 2.545059, dpu acceleration 1.220291
 DPU PRATIO 140.306339, CPU PRATIO 275.865569, DPU OK ? 1 
dpu [ms]: 2.083401, cpu [ms] 2.530594, dpu acceleration 1.214646
 DPU PRATIO 135.292704, CPU PRATIO 271.441071, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


### 

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDICES_PER_LOOKUP=160
MAX_BATCH_SIZE=60
EMBEDDING_DIM=64
BATCH_SIZE=32
NR_RUN=5
MAX_INDICES_PER_LOOKUP=120
MAX_INDICES_PER_LOOKUP_RAND=160
EMBEDDING_DEPTH=500000
NR_EMBEDDING=32
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0


alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 MAX_DPU_EMB_TABLE_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 96
nr cols per dpu 28
alloc dpus 96
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference
max nr embedding 2000
dpu [ms]: 4.991260, cpu [ms] 5.759443, dpu acceleration 1.153906
 DPU PRATIO 140.419561, CPU PRATIO 775.988263, DPU OK ? 1 
dpu [ms]: 4.990955, cpu [ms] 3.463053, dpu acceleration 0.693866
 DPU PRATIO 140.013290, CPU PRATIO 812.253611, DPU OK ? 1 
dpu [ms]: 4.993158, cpu [ms] 3.543014, dpu acceleration 0.709574
 DPU PRATIO 140.068073, CPU PRATIO 802.518922, DPU OK ? 1 
dpu [ms]: 4.992824, cpu [ms] 3.492644, dpu acceleration 0.699533
 DPU PRATIO 140.323319, CPU PRATIO 786.815973, DPU OK ? 1 
dpu [ms]: 4.987660, cpu [ms] 3.456822, dpu acceleration 0.693075
 DPU PRATIO 131.950108, CPU PRATIO 799.585949, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


###

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDICES_PER_LOOKUP=160
MAX_BATCH_SIZE=60
EMBEDDING_DIM=64
BATCH_SIZE=32
NR_RUN=5
MAX_INDICES_PER_LOOKUP=120
MAX_INDICES_PER_LOOKUP_RAND=160
EMBEDDING_DEPTH=500000
NR_EMBEDDING=500
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

./build/emb
alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 MAX_DPU_EMB_TABLE_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 1500
nr cols per dpu 28
alloc dpus 1500
generate synthetic tables


dpu [ms]: 8.423995, cpu [ms] 56.992241, dpu acceleration 6.765465
 DPU PRATIO 800.304310, CPU PRATIO 2904.793451, DPU OK ? 1 
dpu [ms]: 8.044491, cpu [ms] 44.212997, dpu acceleration 5.496059
 DPU PRATIO 784.432158, CPU PRATIO 2983.069806, DPU OK ? 1 
dpu [ms]: 7.952548, cpu [ms] 48.477433, dpu acceleration 6.095837
 DPU PRATIO 791.035259, CPU PRATIO 2904.586238, DPU OK ? 1 
dpu [ms]: 8.022991, cpu [ms] 43.139185, dpu acceleration 5.376946
 DPU PRATIO 784.425146, CPU PRATIO 3018.940376, DPU OK ? 1 
dpu [ms]: 8.058238, cpu [ms] 43.244847, dpu acceleration 5.366539
 DPU PRATIO 754.214129, CPU PRATIO 3007.752643, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)




###

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDICES_PER_LOOKUP=160
MAX_BATCH_SIZE=60
EMBEDDING_DIM=64
BATCH_SIZE=32
NR_RUN=5
MAX_INDICES_PER_LOOKUP=120
MAX_INDICES_PER_LOOKUP_RAND=160
EMBEDDING_DEPTH=500000
NR_EMBEDDING=256
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0


alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 MAX_DPU_EMB_TABLE_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 768
nr cols per dpu 28
alloc dpus 768
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference
max nr embedding 2000
dpu [ms]: 7.521322, cpu [ms] 33.471488, dpu acceleration 4.450213
 DPU PRATIO 422.243293, CPU PRATIO 2758.022371, DPU OK ? 1 
dpu [ms]: 7.547177, cpu [ms] 26.824391, dpu acceleration 3.554228
 DPU PRATIO 418.060201, CPU PRATIO 2745.690948, DPU OK ? 1 
dpu [ms]: 7.481782, cpu [ms] 34.046706, dpu acceleration 4.550614
 DPU PRATIO 416.475137, CPU PRATIO 2548.146408, DPU OK ? 1 
dpu [ms]: 7.688731, cpu [ms] 21.558593, dpu acceleration 2.803921
 DPU PRATIO 431.928749, CPU PRATIO 2841.914560, DPU OK ? 1 
dpu [ms]: 7.456608, cpu [ms] 26.188676, dpu acceleration 3.512143
 DPU PRATIO 396.396212, CPU PRATIO 2609.354646, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


###

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDICES_PER_LOOKUP=160
MAX_BATCH_SIZE=60
EMBEDDING_DIM=64
BATCH_SIZE=50
NR_RUN=5
MAX_INDICES_PER_LOOKUP=120
MAX_INDICES_PER_LOOKUP_RAND=160
EMBEDDING_DEPTH=500000
NR_EMBEDDING=500
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 MAX_DPU_EMB_TABLE_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 1500
nr cols per dpu 28
alloc dpus 1500
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference
max nr embedding 2000
dpu [ms]: 13.234336, cpu [ms] 80.074881, dpu acceleration 6.050540
 DPU PRATIO 772.023810, CPU PRATIO 2920.111227, DPU OK ? 1 
dpu [ms]: 13.105124, cpu [ms] 65.407192, dpu acceleration 4.990963
 DPU PRATIO 780.545537, CPU PRATIO 3030.879925, DPU OK ? 1 
dpu [ms]: 13.195278, cpu [ms] 84.876177, dpu acceleration 6.432314
 DPU PRATIO 770.560487, CPU PRATIO 2656.216090, DPU OK ? 1 
dpu [ms]: 13.167750, cpu [ms] 83.540583, dpu acceleration 6.344332
 DPU PRATIO 781.106934, CPU PRATIO 2631.227533, DPU OK ? 1 
dpu [ms]: 13.122401, cpu [ms] 76.040126, dpu acceleration 5.794681
 DPU PRATIO 744.044440, CPU PRATIO 2773.908096, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

