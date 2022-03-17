# Evaluating the performance impact of word embedding level gender bias mitigation on downstream tasks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheStarterPack/Word-Embedding-Bias-Mitigation-Downstream/blob/main/util/DEBIAS_DOWNSTREAM_EFFECT.ipynb)

### Dependencies:
- Tensorflow 2.8.0
- Hugging Face Datasets
- Gensim 4.1.2
- scikit-learn
- Numpy

### Usage

    python3 main.py [embedding name] [task name] [working directory]/

Choose task name from: {lm, pos, ner, sa, hsd}

Choose embedding name from: {glove, gn-glove, gp-glove, gp-gn-glove} or add your own embedding to embeddings.py.

Expects embeddings in embeddings dir within the working directory, please download from https://github.com/uclanlp/gn_glove and https://github.com/kanekomasahiro/gp_debias/:

	[working directory]/embeddings/glove.txt
	[working directory]/embeddings/gn_glove.txt
	[working directory]/embeddings/gp-glove.txt
	[working directory]/embeddings/gp-gn-glove.txt

Models are saved to the models directory within the working directory:
	
	#the final model after training
	[working directory]/models/[task name]/[embedding name]/model 
 	
	#the model with lowest validation loss
	[working directory]/models/[task name]/[embedding name]/best_model
	
	#tensorboard training logs
	[working directory]/models/[task name]/[embedding name]/logs 



