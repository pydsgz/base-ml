 ### 1 . Prerequisite
* using pip
```
git clone https://github.com/pydsgz/base-ml.git
cd base-ml
pip install -r requirements.txt
```

### 2. Setup dataset
Dataset should be in
```
./data/<dataset_name>/
```

### 3. Train model
```
python train.py --dataset=<dataset_name> --exp_num=1
```
For example, to run experiment using dizzyreg-task 1 dataset. `exp_num` is a 
unique number added to output folders.

```
python train.py --dataset=dizzyreg1 --exp_num=1
```
* All outputs from this experiment will be in
  `./examples/outputs/dizzyreg1/exp_001/`
  
### 4. Plot classification results
You can plot classifiation results using the notebooks in `./notebooks/`
```
cd ./notebooks/
jupyter notebook
```

