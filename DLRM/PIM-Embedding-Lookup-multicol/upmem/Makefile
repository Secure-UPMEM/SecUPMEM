.PHONY:all

# DPUs information
NR_TASKLETS=16

# limits
MAX_EMBEDDING_DIM=64
MAX_NR_EMBEDDING=64
MAX_INDICES_PER_LOOKUP=32
MAX_BATCH_SIZE=128
MAX_INDICES_PER_LOOKUP_RAND=32

# embedding information
EMBEDDING_DIM=64
EMBEDDING_DEPTH=300000
NR_EMBEDDING=64

# inputs information
BATCH_SIZE=128
INDICES_PER_LOOKUP=32
RAND_INPUT_SIZE=0

# benchmark information
NR_RUN=10
CHECK_RESULTS=1

# CPU/DPU check rounding error margin
CHECK_RESULT_ABSOLUTE_ROUNDING_ERROR_MARGIN=5000

build: build/bench build/embdpu

build/bench: build/emblib
	gcc -O3 -fopenmp -Wl,-rpath=build -Wall -mavx -msse4 -lm -lz -lpthread -L build/ -lemb -I include \
		-DINDICES_PER_LOOKUP=${INDICES_PER_LOOKUP} \
		-DEMBEDDING_DIM=${EMBEDDING_DIM} \
		-DEMBEDDING_DEPTH=${EMBEDDING_DEPTH} \
		-DNR_RUN=${NR_RUN} \
		-DBATCH_SIZE=${BATCH_SIZE} \
		-DNR_EMBEDDING=${NR_EMBEDDING} \
		-o build/emb  synthetic_dataset_embedding.c  `dpu-pkg-config --cflags --libs dpu`
    
build/emblib: src/*.c include/*.h
	mkdir -p build
	gcc -O3  -fopenmp -Wall -mavx -msse4 -lm -lz -lpthread -I include \
		-DINDICES_PER_LOOKUP=${INDICES_PER_LOOKUP} \
		-DCHECK_RESULTS=${CHECK_RESULTS} \
		-DRAND_INPUT_SIZE=${RAND_INPUT_SIZE} \
		-DMAX_INDICES_PER_LOOKUP_RAND=${MAX_INDICES_PER_LOOKUP_RAND} \
		-DMAX_NR_EMBEDDING=${MAX_NR_EMBEDDING} \
		-DMAX_BATCH_SIZE=${MAX_BATCH_SIZE} \
		-DCHECK_RESULT_ABSOLUTE_ROUNDING_ERROR_MARGIN=${CHECK_RESULT_ABSOLUTE_ROUNDING_ERROR_MARGIN} \
		-shared -o build/libemb.so -fPIC src/*.c  `dpu-pkg-config --cflags --libs dpu` 

build/embdpu: src/dpu/*.c 
	mkdir -p build
	# TODO : why flto fails
	dpu-clang -O3  -flto=thin -I include \
		-DMAX_EMBEDDING_DIM=${MAX_EMBEDDING_DIM} \
		-DNR_TASKLETS=${NR_TASKLETS} \
		-DMAX_INDICES_PER_LOOKUP=${MAX_INDICES_PER_LOOKUP} \
		-DMAX_BATCH_SIZE=${MAX_BATCH_SIZE} \
		-o build/embdpu src/dpu/dpu_embedding.c

clean:
	([ -d "build" ] && rm -r build/ && rm -f out.cpu.log out.dpu.log) || [ ! -d "build" ]

testdpu: FORCE
	./build/emb

tracedpu: FORCE
	dpu-profiling functions -o dpu.json -a -A \
		-f build_synthetic_input_data \
		-f synthetic_inference \
		--external-function  ./build/libemb.so:lookup \
		--external-function  ./build/libemb.so:populate_mram \
		--external-function  ./build/libemb.so:gather_rank_embedding_results \
		-- ./build/emb

FORCE: ;
