# Limits of the Embedding lookup

as of now:

- `MAX_BATCH_SIZE` couldn't be higher than 512 (2048 bits max in MRAM write). Now unlimited, at the cost of 3% perf
- `EMBEDDING_DEPTH` can't be higher than 14680064 (64 MB per DPU) could be increased by splitting columns on multiple DPUs.
- `MAX_INDICES_PER_LOOKUP` no limit
- `NR_EMBEDDING` no limit
- `EMBEDDING_DIM` no limit
- `NR_EMBEDDING * EMBEDDING_DIM` has to be lower than the number of DPUs.
