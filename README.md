# CLEF2023

This repository contains the code for the  FungiCLEF and SnakeCLEF competitions.

## FungiCLEF2023

dataset is read from the *.csv file in "./metadata" directory, which includes the labels and storage paths of the images.

### usage

1 using the following command to train your own model
    
    python train_seesawloss.py

2 Select the *.pth file from the "./checkpoint" directory and modify the value of the CFG dictionary in "predict.py"

    python predict.py


## SnakeCLEF 2023

dataset is read from the *.csv file in "./metadata" directory, which includes the labels and storage paths of the images.

### usage

1 using the following command to convert the meta text information to a vector

    python extract_metadata_feature.py

2 using the following command to train your own model
    
    python train_pyramid_meta.py

3 using the following command to train your own prior model

    python train_prior.py

4 sellect *.pth file from "./checkpoint" directory and modify the value of the CFG['checkpoints'] and CFG['prior_checkpoint'] in "predict.py". 
Then, using the  the following command to predict

    python predict.py

