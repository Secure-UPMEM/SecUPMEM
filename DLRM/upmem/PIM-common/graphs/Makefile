SRC=../results

all: example.pdf breakdown_example.pdf dpu-transfer.png compression_speedup_gpu.png grep.png aes.png

example.pdf: example.csv ./chart_dpu_vs_host.py
	./chart_dpu_vs_host.py -r $< -o $@ \
		--xlab "Number of DPUs" --ylab "Time (s)"

breakdown_example.pdf: breakdown_example.csv ./chart_breakdown.py
	./chart_breakdown.py -r $< -o $@ \
		--xlab "Raw Filtering Selectivity" --ylab "Time (s)"

dpu-transfer.png: $(SRC)/dpu_data_xfer_speed.csv
	./chart_dpu_data_xfer.py -r $< -o $@

bw_vs_size.png: bw_vs_size.py
	./bw_vs_size.py -r dpu_data_xfer_speed.csv -o $@

compression_speedup_gpu.png: $(SRC)/compression_speedup_gpu.csv aes_dpu_vs_host.py
	./aes_dpu_vs_host.py -r $< -o $@ \
		--xlab "Data size" --ylab "Speedup"

grep.png: $(SRC)/grep.csv
	./chart_dpu_vs_host.py -r $(SRC)/grep.csv -o $@ \
		--xlab "Number of Tasklets" --ylab "Speedup" --speedup

aes.png: $(SRC)/aes.csv chart_dpu_vs_host.py
	./aes_dpu_vs_host.py -r $< -o $@ \
		--xlab "Data size" --ylab "Speedup"
