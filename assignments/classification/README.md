
## Structure of the repo:

- **data**
    
    Datasets and all data-related stuff is here.
    it could be symbolic link.

- **experiments** 

    Folder contains all intermediate results from different experiments
    (param files, checkpoints, logs, ...) 
                
- **notebooks**

    placeholder for jupyter notebooks.
    notebooks could contain some analysis
    (dataset analysis, evalution results), demo, some ongoing work

- **results**

    The useful results are stored here.
    
- **src**

    place for codebase
    
- **README.md**

    markdown readme
    
## How to run

> python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml

or

> python src/main_train.py --param_file experiments/cfgs/params_mnist.yaml

etc