echo " Functionality test"

echo " *********** MLP-Precomputation **********"
cd MLP/MLP-Precomputation/
./run_reproduce.sh

echo " *********** MLP-Runtime **********"
cd ../MLP-Runtime/
./run_reproduce.sh

echo " *********** DLRM-Both **********"
cd ../../DLRM/PIM-Embedding-Lookup-multicol/upmem/
./run_reproduce.sh

echo " *********** Linear Regression training **********"
sudo apt install  -y libgmp-dev
cd ../../../'Linear Regression'/LogReg_int32_float/
./run_reproduce.sh

echo " *********** Logistic Regression training (A) **********"
cd ../../'Logistic Regression'/LogReg_Runtime_A/
./run_reproduce.sh

echo " *********** Logistic Regression training (A2Y) **********"
cd ../../'Logistic Regression'/LogReg_Runtime_A2Y/
./run_reproduce.sh

