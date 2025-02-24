# Define datasets
Generate_Set = 'generate/'
Raw_Set = 'raw/'
# Check environment, if no torchvision, install it
if not 'torchvision' in sys.modules:
    !pip install torchvision
# Check environment, if no torch, install it
if not 'torch' in sys.modules:
    !pip install torch
# Check environment, if no pytorch_fid, install it
if not 'pytorch_fid' in sys.modules:
    !pip install pytorch-fid
# Get terminal output of FID evaluation (get the last line and save as a file)
!python -m pytorch_fid {Raw_Set} {Generate_Set} > "../fid_output.txt"

