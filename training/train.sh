#eyeglass
python train.py --max_steps 10 --changes 15 --keeps 20 -1 --run_name 15
#gender
python train.py --max_steps 10 --changes 20 --keeps 15 -1 --run_name 20
#age
python train.py --max_steps 10 --changes -1 --keeps 15 20 --run_name Age
#smile
python train.py --max_steps 10 --changes 31 --keeps 15 20 -1 --run_name 31
#young
python train.py --max_steps 10 --changes 39 --keeps 15 20 --run_name 39

#hair color
python train_onehot.py --max_steps 10 --changes 8 9 11 --keeps 15 20 31 -1 --run_name 8_9_11
#hair type
python train_onehot.py --max_steps 10 --changes 32 33 --keeps 15 20 31 -1 --run_name 32_33