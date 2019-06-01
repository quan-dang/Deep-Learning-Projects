# Neural Machine Translation
Build Machine Learning Translation using LSTMs.

## Configuration
Verifed on GTX 1016 16GB

## How to run
Run MachineTranslationOneHot.py for one-hot encoded 

'''
python MachineTranslationOneHot.py --path 'data/fra.txt' --epochs 20 --batch_size 32 -latent_dim 128 --num_samples 10000 --outdir 'outputs' --verbose 1 --mode train
'''

__OR__ 

Run MachineTranslationWord2Vec.py for word vector embeddings

'''
python MachineTranslationWord2Vec.py --path 'data/fra.txt' --epochs 20 --batch_size 32 -latent_dim 128 --num_samples 40000 --outdir 'outputs' --verbose 1 --mode train --embedding_dim 128
'''

## Datasets
[Data] (http://www.manythings.org/anki/)

Dataset name : fra-eng/fra.txt