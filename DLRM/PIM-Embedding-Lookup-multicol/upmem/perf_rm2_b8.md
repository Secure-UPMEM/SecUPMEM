gcc -O3     -Wall -mavx -msse4 -lm -lz -lpthread -I include \
	-DNR_RUN=2 \
	-DCHECK_RESULTS=1 \
	-DMAX_NR_DPUS=10000 \
	-DRAND_INPUT_SIZE=0 \
	-DMAX_INDICES_PER_LOOKUP_RAND=160 \
	-DNR_TASKLETS=16 \
	-DMAX_INDICES_PER_LOOKUP=160 \
	-DMAX_NR_EMBEDDING=2000 \
	-DMAX_BATCH_SIZE=60 \
	-DMAX_INDICES_PER_LOOKUP=120 \
	-DEMBEDDING_DEPTH=500000 \
	-DBATCH_SIZE=8 \
	-DNR_EMBEDDING=500 \
	-DEMBEDDING_DIM=64 \
	-o build/emb src/*.c  `dpu-pkg-config --cflags --libs dpu` 
mkdir -p build
# TODO : why flto fails
dpu-clang -O3  -flto=thin -I include \
	-DNR_RUN=2 \
	-DNR_TASKLETS=16 \
	-DMAX_NR_DPUS=10000 \
	-DRAND_INPUT_SIZE=0 \
	-DMAX_INDICES_PER_LOOKUP=160 \
	-DMAX_NR_EMBEDDING=2000 \
	-DMAX_BATCH_SIZE=60 \
	-DNR_EMBEDDING=500 \
	-DMAX_INDICES_PER_LOOKUP=120 \
	-DEMBEDDING_DEPTH=500000 \
	-DBATCH_SIZE=8 \
	-o build/embdpu src/dpu/dpu_embedding.c
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
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference
max nr embedding 2000
[PERF] DPU CLOCK RAW time [ms]:  5.78
[PERF] DPUTIME time [ms]: 46.54
[PERF] DPU PROCESS ratio: 804.75
[PERF] DPU CLOCK RAW time [ms]:  4.54
[PERF] DPUTIME time [ms]: 32.78
[PERF] DPU PROCESS ratio: 722.76
[PERF] DPU CLOCK RAW time [ms]:  4.43
[PERF] DPUTIME time [ms]: 32.31
[PERF] DPU PROCESS ratio: 729.56
[PERF] DPU CLOCK RAW time [ms]:  4.23
[PERF] DPUTIME time [ms]: 31.69
[PERF] DPU PROCESS ratio: 748.65
[PERF] DPU CLOCK RAW time [ms]:  4.04
[PERF] DPUTIME time [ms]: 27.37
[PERF] DPU PROCESS ratio: 676.63
[PERF] DPU CLOCK RAW time [ms]:  3.92
[PERF] DPUTIME time [ms]: 27.53
[PERF] DPU PROCESS ratio: 702.08
[PERF] DPU CLOCK RAW time [ms]:  4.36
[PERF] DPUTIME time [ms]: 28.48
[PERF] DPU PROCESS ratio: 652.48
[PERF] DPU CLOCK RAW time [ms]:  4.40
[PERF] DPUTIME time [ms]: 28.11
[PERF] DPU PROCESS ratio: 638.28
[PERF] DPU CLOCK RAW time [ms]:  4.16
[PERF] DPUTIME time [ms]: 27.84
[PERF] DPU PROCESS ratio: 668.51
[PERF] DPU CLOCK RAW time [ms]:  4.12
[PERF] DPUTIME time [ms]: 27.38
[PERF] DPU PROCESS ratio: 664.90
[PERF] DPU CLOCK RAW time [ms]:  4.48
[PERF] DPUTIME time [ms]: 28.84
[PERF] DPU PROCESS ratio: 643.80
[PERF] DPU CLOCK RAW time [ms]:  4.39
[PERF] DPUTIME time [ms]: 28.78
[PERF] DPU PROCESS ratio: 655.64
[PERF] DPU CLOCK RAW time [ms]:  4.29
[PERF] DPUTIME time [ms]: 28.66
[PERF] DPU PROCESS ratio: 668.23
[PERF] DPU CLOCK RAW time [ms]:  4.33
[PERF] DPUTIME time [ms]: 28.71
[PERF] DPU PROCESS ratio: 663.74
[PERF] DPU CLOCK RAW time [ms]:  4.39
[PERF] DPUTIME time [ms]: 28.33
[PERF] DPU PROCESS ratio: 644.96
[PERF] DPU CLOCK RAW time [ms]:  4.41
[PERF] DPUTIME time [ms]: 28.45
[PERF] DPU PROCESS ratio: 645.54
[PERF] DPU CLOCK RAW time [ms]:  4.17
[PERF] DPUTIME time [ms]: 27.99
[PERF] DPU PROCESS ratio: 671.50
[PERF] DPU CLOCK RAW time [ms]:  4.18
[PERF] DPUTIME time [ms]: 28.43
[PERF] DPU PROCESS ratio: 680.34
[PERF] DPU CLOCK RAW time [ms]:  4.17
[PERF] DPUTIME time [ms]: 28.60
[PERF] DPU PROCESS ratio: 686.35
[PERF] DPU CLOCK RAW time [ms]:  4.13
[PERF] DPUTIME time [ms]: 28.30
[PERF] DPU PROCESS ratio: 684.60
[PERF] CPU CLOCK RAW time [ms]: 74.37
[PERF] CPUTIME time [ms]: 294.90
[PERF] CPU PROCESS ratio: 396.53
[PERF] CPU CLOCK RAW time [ms]: 11.82
[PERF] CPUTIME time [ms]: 312.14
[PERF] CPU PROCESS ratio: 2641.68
[PERF] CPU CLOCK RAW time [ms]: 11.29
[PERF] CPUTIME time [ms]: 313.27
[PERF] CPU PROCESS ratio: 2774.89
[PERF] CPU CLOCK RAW time [ms]: 11.50
[PERF] CPUTIME time [ms]: 312.03
[PERF] CPU PROCESS ratio: 2713.87
[PERF] CPU CLOCK RAW time [ms]: 11.14
[PERF] CPUTIME time [ms]: 305.62
[PERF] CPU PROCESS ratio: 2743.66
[PERF] CPU CLOCK RAW time [ms]: 10.98
[PERF] CPUTIME time [ms]: 301.44
[PERF] CPU PROCESS ratio: 2746.17
[PERF] CPU CLOCK RAW time [ms]: 11.00
[PERF] CPUTIME time [ms]: 304.27
[PERF] CPU PROCESS ratio: 2765.05
[PERF] CPU CLOCK RAW time [ms]: 10.78
[PERF] CPUTIME time [ms]: 303.66
[PERF] CPU PROCESS ratio: 2817.91
[PERF] CPU CLOCK RAW time [ms]: 10.77
[PERF] CPUTIME time [ms]: 303.18
[PERF] CPU PROCESS ratio: 2816.05
[PERF] CPU CLOCK RAW time [ms]: 10.93
[PERF] CPUTIME time [ms]: 305.11
[PERF] CPU PROCESS ratio: 2792.71
[PERF] CPU CLOCK RAW time [ms]: 10.76
[PERF] CPUTIME time [ms]: 298.89
[PERF] CPU PROCESS ratio: 2777.81
[PERF] CPU CLOCK RAW time [ms]: 10.71
[PERF] CPUTIME time [ms]: 300.46
[PERF] CPU PROCESS ratio: 2804.45
[PERF] CPU CLOCK RAW time [ms]: 10.71
[PERF] CPUTIME time [ms]: 298.66
[PERF] CPU PROCESS ratio: 2787.34
[PERF] CPU CLOCK RAW time [ms]: 10.72
[PERF] CPUTIME time [ms]: 300.23
[PERF] CPU PROCESS ratio: 2801.06
[PERF] CPU CLOCK RAW time [ms]: 10.83
[PERF] CPUTIME time [ms]: 301.89
[PERF] CPU PROCESS ratio: 2788.60
[PERF] CPU CLOCK RAW time [ms]: 10.72
[PERF] CPUTIME time [ms]: 299.93
[PERF] CPU PROCESS ratio: 2798.10
[PERF] CPU CLOCK RAW time [ms]: 10.84
[PERF] CPUTIME time [ms]: 301.73
[PERF] CPU PROCESS ratio: 2784.34
[PERF] CPU CLOCK RAW time [ms]: 10.82
[PERF] CPUTIME time [ms]: 301.13
[PERF] CPU PROCESS ratio: 2781.94
[PERF] CPU CLOCK RAW time [ms]: 10.75
[PERF] CPUTIME time [ms]: 298.52
[PERF] CPU PROCESS ratio: 2777.01
[PERF] CPU CLOCK RAW time [ms]: 10.70
[PERF] CPUTIME time [ms]: 298.17
[PERF] CPU PROCESS ratio: 2787.75
dpu [ms]: 4.346306, cpu [ms] 14.105919, dpu acceleration 3.245496
 DPU PRATIO 682.664760, CPU PRATIO 2654.846506, DPU OK ? 1 
[PERF] DPU CLOCK RAW time [ms]:  3.56
[PERF] DPUTIME time [ms]: 21.90
[PERF] DPU PROCESS ratio: 614.82
[PERF] DPU CLOCK RAW time [ms]:  3.13
[PERF] DPUTIME time [ms]: 18.78
[PERF] DPU PROCESS ratio: 599.74
[PERF] DPU CLOCK RAW time [ms]:  3.18
[PERF] DPUTIME time [ms]: 19.14
[PERF] DPU PROCESS ratio: 601.33
[PERF] DPU CLOCK RAW time [ms]:  3.22
[PERF] DPUTIME time [ms]: 19.03
[PERF] DPU PROCESS ratio: 591.35
[PERF] DPU CLOCK RAW time [ms]:  3.46
[PERF] DPUTIME time [ms]: 20.77
[PERF] DPU PROCESS ratio: 600.19
[PERF] DPU CLOCK RAW time [ms]:  3.63
[PERF] DPUTIME time [ms]: 22.84
[PERF] DPU PROCESS ratio: 628.97
[PERF] DPU CLOCK RAW time [ms]:  3.53
[PERF] DPUTIME time [ms]: 22.91
[PERF] DPU PROCESS ratio: 648.67
[PERF] DPU CLOCK RAW time [ms]:  4.05
[PERF] DPUTIME time [ms]: 25.19
[PERF] DPU PROCESS ratio: 621.63
[PERF] DPU CLOCK RAW time [ms]:  4.39
[PERF] DPUTIME time [ms]: 27.93
[PERF] DPU PROCESS ratio: 636.64
[PERF] DPU CLOCK RAW time [ms]:  4.10
[PERF] DPUTIME time [ms]: 27.29
[PERF] DPU PROCESS ratio: 665.61
[PERF] DPU CLOCK RAW time [ms]:  4.14
[PERF] DPUTIME time [ms]: 27.66
[PERF] DPU PROCESS ratio: 668.43
[PERF] DPU CLOCK RAW time [ms]:  4.16
[PERF] DPUTIME time [ms]: 28.00
[PERF] DPU PROCESS ratio: 672.97
[PERF] DPU CLOCK RAW time [ms]:  4.20
[PERF] DPUTIME time [ms]: 27.90
[PERF] DPU PROCESS ratio: 663.98
[PERF] DPU CLOCK RAW time [ms]:  4.23
[PERF] DPUTIME time [ms]: 28.43
[PERF] DPU PROCESS ratio: 672.74
[PERF] DPU CLOCK RAW time [ms]:  4.17
[PERF] DPUTIME time [ms]: 28.08
[PERF] DPU PROCESS ratio: 673.64
[PERF] DPU CLOCK RAW time [ms]:  3.84
[PERF] DPUTIME time [ms]: 27.35
[PERF] DPU PROCESS ratio: 712.10
[PERF] DPU CLOCK RAW time [ms]:  4.07
[PERF] DPUTIME time [ms]: 27.59
[PERF] DPU PROCESS ratio: 678.08
[PERF] DPU CLOCK RAW time [ms]:  4.16
[PERF] DPUTIME time [ms]: 27.32
[PERF] DPU PROCESS ratio: 656.23
[PERF] DPU CLOCK RAW time [ms]:  4.19
[PERF] DPUTIME time [ms]: 27.96
[PERF] DPU PROCESS ratio: 667.79
[PERF] DPU CLOCK RAW time [ms]:  4.37
[PERF] DPUTIME time [ms]: 28.92
[PERF] DPU PROCESS ratio: 662.27
[PERF] CPU CLOCK RAW time [ms]: 13.63
[PERF] CPUTIME time [ms]: 282.95
[PERF] CPU PROCESS ratio: 2075.29
[PERF] CPU CLOCK RAW time [ms]: 11.40
[PERF] CPUTIME time [ms]: 309.02
[PERF] CPU PROCESS ratio: 2711.66
[PERF] CPU CLOCK RAW time [ms]: 11.45
[PERF] CPUTIME time [ms]: 307.87
[PERF] CPU PROCESS ratio: 2689.81
[PERF] CPU CLOCK RAW time [ms]: 11.02
[PERF] CPUTIME time [ms]: 301.94
[PERF] CPU PROCESS ratio: 2740.29
[PERF] CPU CLOCK RAW time [ms]: 11.09
[PERF] CPUTIME time [ms]: 309.05
[PERF] CPU PROCESS ratio: 2787.70
[PERF] CPU CLOCK RAW time [ms]: 11.18
[PERF] CPUTIME time [ms]: 300.82
[PERF] CPU PROCESS ratio: 2689.67
[PERF] CPU CLOCK RAW time [ms]: 11.04
[PERF] CPUTIME time [ms]: 303.90
[PERF] CPU PROCESS ratio: 2752.82
[PERF] CPU CLOCK RAW time [ms]: 10.89
[PERF] CPUTIME time [ms]: 300.36
[PERF] CPU PROCESS ratio: 2759.32
[PERF] CPU CLOCK RAW time [ms]: 10.92
[PERF] CPUTIME time [ms]: 304.57
[PERF] CPU PROCESS ratio: 2788.60
[PERF] CPU CLOCK RAW time [ms]: 10.72
[PERF] CPUTIME time [ms]: 299.27
[PERF] CPU PROCESS ratio: 2791.94
[PERF] CPU CLOCK RAW time [ms]: 10.74
[PERF] CPUTIME time [ms]: 301.22
[PERF] CPU PROCESS ratio: 2804.48
[PERF] CPU CLOCK RAW time [ms]: 10.73
[PERF] CPUTIME time [ms]: 298.45
[PERF] CPU PROCESS ratio: 2781.33
[PERF] CPU CLOCK RAW time [ms]: 10.89
[PERF] CPUTIME time [ms]: 304.08
[PERF] CPU PROCESS ratio: 2792.56
[PERF] CPU CLOCK RAW time [ms]: 10.87
[PERF] CPUTIME time [ms]: 301.81
[PERF] CPU PROCESS ratio: 2777.30
[PERF] CPU CLOCK RAW time [ms]: 10.75
[PERF] CPUTIME time [ms]: 300.03
[PERF] CPU PROCESS ratio: 2792.16
[PERF] CPU CLOCK RAW time [ms]: 10.77
[PERF] CPUTIME time [ms]: 300.51
[PERF] CPU PROCESS ratio: 2790.43
[PERF] CPU CLOCK RAW time [ms]: 10.73
[PERF] CPUTIME time [ms]: 298.17
[PERF] CPU PROCESS ratio: 2778.34
[PERF] CPU CLOCK RAW time [ms]: 10.71
[PERF] CPUTIME time [ms]: 298.47
[PERF] CPU PROCESS ratio: 2786.79
[PERF] CPU CLOCK RAW time [ms]: 10.75
[PERF] CPUTIME time [ms]: 298.10
[PERF] CPU PROCESS ratio: 2772.33
[PERF] CPU CLOCK RAW time [ms]: 10.90
[PERF] CPUTIME time [ms]: 298.39
[PERF] CPU PROCESS ratio: 2736.55
dpu [ms]: 3.888967, cpu [ms] 11.058569, dpu acceleration 2.843575
 DPU PRATIO 646.859161, CPU PRATIO 2729.969225, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

