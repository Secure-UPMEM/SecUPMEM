# SecUPMEM

[![DOI](https://zenodo.org/badge/921982568.svg)](https://doi.org/10.5281/zenodo.14736863)


Enabling Low-Cost Secure Computing on Untrusted In-Memory Architectures

Note: This implementation is intended solely for performance experiments.

**Code references:**

Below are the links to our (both CPU-based and UPMEM-based) baselines. Our code is built upon these implementations, and we compare our implementation with them:

MLP: 

https://github.com/CMU-SAFARI/prim-benchmarks

DLRM: 

https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup

https://github.com/upmem/PIM-Embedding-Lookup/tree/multicol/upmem

Logistic Regression and Linear Regression: 

(Training:) https://github.com/CMU-SAFARI/pim-ml

(Linear Regression inference: ) https://github.com/CMU-SAFARI/prim-benchmarks

AES Implementation: 

https://github.com/kokke/tiny-AES-c/blob/master/aes.c


To run a simple functionality check, execute:
```
./run_functionality.sh
```

To regenerate our results execute:
```
./run_reproduce.sh
```


The run.sh file builds the host and DPU-side code, links them, and finally runs the program with our configurations.

## Setup
For all applications, you need to install the UPMEM SDK: https://sdk.upmem.com/
Note: The UPMEM SDK provides a simulator that you can use to run our code without requiring access to the real hardware. However, the simulator cannot be used to reproduce our performance results.

For each application, please refer to the respective baseline repository for complete instructions on how to set up the environment.
