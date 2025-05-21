# Secure UPMEM

This is the open-source implementation of the proposed method in the paper:

[Enabling Low-Cost Secure Computing on Untrusted In-Memory Architectures](https://arxiv.org/abs/2501.17292)

This work leverages multi-party computation (MPC) techniques, specifically arithmetic secret sharing and Yao’s garbled circuits, to outsource bandwidth-intensive computation securely to PIM. Additionally, we leverage precomputation optimization to prevent the CPU's portion of the MPC from becoming a bottleneck. We evaluate our approach using the UPMEM PIM system over various applications. We provided all the source codes and scripts for evaluating our scheme using UPMEM, the first publicly available PIM, over four data-intensive applications: Multilayer Perceptron inference (MLP), Deep Learning Recommendation Model inference (DLRM), linear regression training, and logistic regression training. This artifact allows researchers to reproduce our results, explore this area further, and expand our work. Our evaluations demonstrate up to a 14.66× speedup compared to a secure CPU configuration while maintaining data confidentiality and integrity when outsourcing linear and/or nonlinear computation.

Note: This implementation is intended solely for performance experiments.

## Setup
For all applications, you need to install the UPMEM SDK: https://sdk.upmem.com/

**Ubuntu Linux**

```
$ git clone https://github.com/Secure-UPMEM/SecUPMEM.git
$ cd SecUPMEM
```

Note: The UPMEM SDK provides a simulator that you can use to run our code without requiring access to the real hardware. However, the simulator cannot be used to reproduce our performance results.

For each application, please refer to the respective baseline repository for complete instructions on how to set up the environment.


**Code references:**

Below are the links to our (both CPU-based and UPMEM-based) baselines. Our code is built upon these implementations, and we compare our implementation with them:

MLP: 

https://github.com/CMU-SAFARI/prim-benchmarks

DLRM: 

https://github.com/UBC-ECE-Sasha/PIM-Embedding-Lookup

https://github.com/upmem/PIM-Embedding-Lookup/tree/multicol/upmem

Logistic Regression and Linear Regression Training: 

https://github.com/CMU-SAFARI/pim-ml

Linear Regression inference:

https://github.com/CMU-SAFARI/prim-benchmarks

AES Implementation: 

https://github.com/kokke/tiny-AES-c/blob/master/aes.c

## Usage

To run a simple functionality check, execute:
```
./run_functionality.sh
```

To regenerate our results, execute:
```
./run_reproduce.sh
```

The comments in the ./run_reproduce.sh file explain which command corresponds to which figure.


The run.sh file builds the host and DPU-side code, links them, and finally runs the program with our configurations.

## Dataset
Our scheme is evaluated using MLP inference, DLRM inference, logistic regression training, and linear regression training. To evaluate our implementation, we use randomly generated inputs.

## Issues

Please refer to https://github.com/Secure-UPMEM/SecUPMEM/issues for a list of known issues or to report a new one.

## Citation

Please cite the following paper if you find this repository useful.

**Archive version:**

S. G. Ghinani, J. Zhang, and E. Sadredini, “Enabling low-cost secure computing on untrusted in-memory architectures,” arXiv preprint arXiv:2501.17292, 2025.

```
@misc{ghinani2025enablinglowcostsecurecomputing,
      title={Enabling Low-Cost Secure Computing on Untrusted In-Memory Architectures}, 
      author={Sahar Ghoflsaz Ghinani and Jingyao Zhang and Elaheh Sadredini},
      year={2025},
      eprint={2501.17292},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2501.17292}, 
}
```

**USENIX Security 2025 version:**

Our work has been accepted to USENIX Security 2025. More materials related to this submission, such as the presentation and artifact description, can be found here: https://www.usenix.org/conference/usenixsecurity25/presentation/ghinani

[![DOI](https://zenodo.org/badge/921982568.svg)](https://doi.org/10.5281/zenodo.14736863)

## Acknowledgements

This work is funded, in part, by NSF Career Award #2339317,
NSF #2235398, the initial complement fund from UCR, and
the Hellman Fellowship from the University of California.
We appreciate the use of hardware resources provided by
the UPMEM. We thank Prof. Nael Abu-Ghazaleh for his
insightful comments.
