# Chart Generators

## Barchart

## Usage

```txt
usage: barchart.py [-h] --csvfile CSVFILE [--xlabel XLABEL] [--ylabel YLABEL] [--xstepsize XSTEPSIZE] [--xstart XSTART]
                   [--xstop XSTOP] [--ystepsize YSTEPSIZE] [--title TITLE] [--outputfile OUTPUTFILE] --headers HEADERS
                   [HEADERS ...]

Generate a bar chart

optional arguments:
  -h, --help            show this help message and exit
  --csvfile CSVFILE     Input CSV data file
  --xlabel XLABEL       Chart x-axis label
  --ylabel YLABEL       Chart y-axis label
  --xstepsize XSTEPSIZE
                        Chart x-axis stepsize
  --xstart XSTART       x-axis start position
  --xstop XSTOP         x-axis stop position
  --ystepsize YSTEPSIZE
                        Chart y-axis stepsize
  --title TITLE         Chart title
  --outputfile OUTPUTFILE
                        Output chart
  --headers HEADERS [HEADERS ...]
                        Headers

```

## Examples

Bar:

```shell script
./barchart --csvfile barchart.csv.example \
    --xstart 192 --xstop 576 --xstepsize 64 --headers Prepare "Copy in" Run NR_DPUS
```

[Example output](barchart.example.pdf)
