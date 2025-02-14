echo " Regeneration of Figures 13 to 18 "

# Fig 13 & 14
cd MLP/MLP-Precomputation/
./run_reproduce.sh

#Fig 13 & 14
cd ../MLP-Runtime/
./run_reproduce.sh

#Fig 15
cd ../../DLRM/PIM-Embedding-Lookup-multicol/upmem/
./run_reproduce.sh

#Fig 17
cd ../../../'Linear Regression'/inference/GEMVFull/
./run_reproduce.sh

#Fig 18
cd ../../../'Linear Regression'/LogReg_int32_float/
./run_reproduce.sh

#Fig 16
cd ../../'Logistic Regression'/LogReg_Runtime_A/
./run_reproduce.sh

#Fig 16
cd ../../'Logistic Regression'/LogReg_Runtime_A2Y/
./run_reproduce.sh

