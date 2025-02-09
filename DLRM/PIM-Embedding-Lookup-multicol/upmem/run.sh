#!/bin/sh

set -e

build=false
run=false
debug=false
verbose=false

usage() {
    printf "%s %s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n" \
        "USAGE:" "${0}" \
        "[ -b ] - Build code" \
        "[ -r ] - Run code" \
        "[ -d ] - Debug" \
        "[ -V ] - Verbose" \
        "<DATASET - kaggle | random | toy>"
    exit 0
}

die() {
    echo "${1}" 1>&2
    exit 1
}

dataset_valid() {
    [ "${1}" = "kaggle" ] && return 0
    [ "${1}" = "random" ] && return 0
    [ "${1}" = "toy" ] && return 0

    return 1
}

kaggle_env() {
    export NR_EMBEDDING=26
    export EMBEDDING_DIM=16
    export MAX_BATCH_SIZE=512
    export NR_TASKLETS=14
}

random_env() {
    export NR_EMBEDDING=9
    export EMBEDDING_DIM=64
    export MAX_BATCH_SIZE=64
    export NR_TASKLETS=14
}

# Old - pre-loadgenerator
# toy_env() {
#     export NR_EMBEDDING=1
#     export EMBEDDING_DIM=8
#     export DPU_TEST=1
#     export NR_TASKLETS=1
# }
toy_env() {
    export NR_EMBEDDING=9
    export EMBEDDING_DIM=64
    export BATCH_SIZE=64
    export DPU_TEST=1
    export MAX_BATCH_SIZE=64
    export NR_TASKLETS=10
    # rows?
}

global_env() {
    "${verbose}" && export V=1
    "${debug}" && export DEBUG=1
    return 0
}

kaggle_run() {
    dlrm="${cwd}/../dlrm"
    python "${dlrm}/dlrm_dpu_pytorch.py" \
           --arch-sparse-feature-size=16 \
           --arch-mlp-bot="13-512-256-64-16" \
           --arch-mlp-top="512-256-1" \
           --data-generation=dataset \
           --data-set=kaggle \
           --processed-data-file="${dlrm}/raw_data/kaggleAdDisplayChallenge_processed.npz" \
           --load-model="${dlrm}/trainedModels/kaggle-model-graham-final.pt" \
           --mini-batch-size=500 \
           --nepochs=1 \
           --inference-only
}

random_run() {
    python3 "${cwd}/../PIM-dlrm-new/dlrm_dpu_pytorch.py" \
           --arch-embedding-size=65000-65000-65000-65000-65000-65000-65000-65000-65000 \
           --arch-sparse-feature-size=64 \
           --arch-mlp-bot=1440-720-64 \
           --arch-mlp-top=40-20-10-1 \
           --data-generation=random \
           --mini-batch-size=64 \
           --num-batches=100 \
           --num-indices-per-lookup=32 \
           --num-indices-per-lookup-fixed=True \
           --inference-only
}

toy_run() {
    echo "DPU_TEST=${DPU_TEST}"
    ./emb_host
}

build_code() {
    # Set ENV
    "${dataset}_env"
    global_env

    make
}

run_code() {
    cwd="${PWD}"
    if "${debug}"; then
        build_dir="./build/debug/"
    else
        build_dir="./build/release/"
    fi
    cd "${build_dir}"
    "${dataset}_run"
}

main() {
    "${build}" && build_code
    "${run}" && run_code
}

options=':bVrdh'
while getopts "${options}" option; do
    case $option in
        b  ) build=true;;
        r  ) run=true;;
        d  ) debug=true;;
        V  ) verbose=true;;
        h  ) usage;;
        \? ) echo "Unknown option: -${OPTARG}" >&2; exit 1;;
        :  ) echo "Missing option argument for -${OPTARG}" >&2; exit 1;;
        *  ) echo "Unimplemented option: -${OPTARG}" >&2; exit 1;;
    esac
done

shift $((OPTIND - 1))

if ! dataset_valid "${1}"; then
    die "'${1}' is not a valid dataset (kaggle | random | toy)"
fi

if ! "${build}" && ! "${run}"; then
    die "Neither -b or -r were used, doing nothing"
    exit 0
fi

dataset="${1}"

main
