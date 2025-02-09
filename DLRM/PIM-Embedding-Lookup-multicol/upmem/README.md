## dpu embedding

Embedding lookup table running on UPMEM PIM DPU.
The design provides a library emblib.so and a testbench that perform
functional validation and performances measurement (plus CPU backend comparaision)
of the embedding lookup table DPU implementation, with synthetic table and synthetic
input datas. 

## testbench flow

```
     ____________________________________________
    |   ______________                           |
    |  |              |           (1)            |
    |  | synthetic    |      intialization       | 
    |  | lookup table |                          |
    |  | generation   |      _______________     |
    |  |______________|---->|               |    |
    |   ___________         | populate DPUs |    |
    |  |           |        | MRAM          |    |
    |  | embedding |------->|_______________|    |
    |  | tables    |                             |
    |  | DPU pool  |                             |
    |  | decoded   |                             |
    |  |___________|                             |
    |____________________________________________|

     ____________________________________________
    |   ______________                           |
    |  |              |           (2)            |
    |  | synthetic    |        inference         | 
    |  | input batch  |                          |
    |  | genereatio   |      _______________     |
    |  |______________|---->|               |    |
    |   _____________    |  | DPU lookup    |    |
    |  |             |   |  |               |    |
    |  | funtional   |<-----|_______________|    |
    |  | validation  |   |                       |
    |  |    +        |   |                       |
    |  | perf        |   |    _______________    |
    |  | measurement |   |-->|               |   |
    |  |_____________|<------| CPU lookup    |   |
    |                        |_______________|   | 
    |____________________________________________|

```
## build
```shell
make clean && make
```
## run
```shell
make testdpu
```
