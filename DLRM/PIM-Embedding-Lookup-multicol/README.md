# PIM-Embedding-Lookup
Off loading embedding lookups to in-memory-processing on UPMEM memories

# Loading the model just for inference with 1 batch

python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --processed-data-file="tests/kaggleAdDisplayChallenge_processed.npz" --mini-batch-size=1 --print-time --load-model="trainedModels/model.pt" --inference-only

# Training the model and saving it

python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --processed-data-file="tests/kaggleAdDisplayChallenge_processed.npz" --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=188 --test-freq=209 --print-time --nepochs=20 --save-model="trainedModels/model.pt"

# Loading the model just for inference

python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --processed-data-file="tests/kaggleAdDisplayChallenge_processed.npz" --mini-batch-size=39292 --test-freq=2 --print-time --load-model="trainedModels/model.pt" --print-freq=1 --nepochs=10 --inference-only
