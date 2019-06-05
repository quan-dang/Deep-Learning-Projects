# Maria Detection using ResNet

## Dataset
The project use [Maria Datasets](https://ceb.nlm.nih.gov/repositories/malaria-datasets/) from U.S National Library of Medicine.

The dataset consists of 27,588 images belonging to 2 seperate classes (13,794 for each):

1. __Parasitized:__ the region containes malaria

2. __Uninfected:__ no evidence of malaria in the region

## Project structure

* malaria/ 
    - cell_images/
        - Parasitized
        - Uninfected
* utilities/
* build_dataset.py
* train_model.py

## How to run

1. __Step 1:__ Preparation

Download the dataset and extract it into folder __malaria/__ in the root path of the project, and rename is as __cell_images/__.

2. __Step 2__: Build the dataset

```
python build_dataset.py
```

3. __Step 3__: Train and evaluate the model

```
python train_model.py
```

