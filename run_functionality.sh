echo " Functionality test"


cd MLP/MLP-Precomputation/
./run_functionality.sh


cd ../MLP-Runtime/
./run_functionality.sh


cd ../../DLRM/PIM-Embedding-Lookup-multicol/upmem/
./run_functionality.sh


cd ../../../'Linear Regression'/inference/GEMVFull/
./run_functionality.sh


cd ../../../'Linear Regression'/LogReg_int32_float/
./run_functionality.sh


cd ../../'Logistic Regression'/LogReg_Runtime_A/
./run_functionality.sh

cd ../../'Logistic Regression'/LogReg_Runtime_A2Y/
./run_functionality.sh

