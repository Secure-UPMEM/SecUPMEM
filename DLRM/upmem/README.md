# PIM

## Building

There are many custom parameters that can be set at compile time in `Makefile`.

For a "default" build, just run `make`

The compiled code will be located in `build/release`, and expects to be run from that directory.

When a build is done with `DEBUG=1`, the code will be located in `build/debug`.

## Convenience run and build script

Rather than setting parameters every time, based on the data set used, the `run.sh` script can be used to build and/or run the code.

```txt
USAGE: ./run.sh
        [ -b ] - Build code
        [ -r ] - Run code
        [ -d ] - Debug
        [ -V ] - Verbose
        <DATASET - kaggle | random | toy>
```

The code can be compiled or run, or both at the same time (`-r` assumes you already built).

```shell script
./run.sh -b toy      # build toy with debug
./run.sh -r toy      # run toy
./run.sh -brdd toy   # build and run toy with debug
```
