echo " Functionality test"

echo " *********** MLP-Precomputation **********"
cd MLP/MLP-Precomputation/
./run_functionality.sh

echo " *********** MLP-Runtime **********"
cd ../MLP-Runtime/
./run_functionality.sh

echo " *********** DLRM-Both **********"
cd ../../DLRM/PIM-Embedding-Lookup-multicol/upmem/
./run_functionality.sh

echo " *********** Linear Regression training **********"
cd ../../../'Linear Regression'/LogReg_int32_float/
./run_functionality.sh

echo " *********** Logistic Regression training (A) **********"
cd ../../'Logistic Regression'/LogReg_Runtime_A/
./run_functionality.sh

echo " *********** Logistic Regression training (A2Y) **********"
cd ../../'Logistic Regression'/LogReg_Runtime_A2Y/
./run_functionality.sh

