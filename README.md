 1. Make sure we are using the same package versions. 
 * using docker: pull our docker image from dockerhub
```
docker pull pydsgz/base_ml:latest
```
* using pip
```
pip install -r requirements.txt
```

2. Check if dataset is available.
- For dizzyreg experiment, make sure dataset is in `/data/dizzyreg/` folder


3. Training mode
* for dizzyreg task 1
```
python train_dizzyreg.py --dataset=dizzyreg1 --exp_num=1
```
