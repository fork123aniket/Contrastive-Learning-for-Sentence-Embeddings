# Contrastive-Learning-for-Sentence-Embeddings
This repository provides a simple code to implement unsupervised contrastive learning framework for generating sentence embeddings from the paper "[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)".
## Setup Environment Requirements
- `Python 3.9`
- `transformers 4.16.2`
- `tensorflow 2.8`
- `tensorflow-addons 0.16.1`
- `scikit-learn 1.0.2`
- `numpy 1.22.2`
- `pandas 1.4.1`
## Usage
### Data
Dataset used can be downloaded directly by using the command, written in ***data/data.txt*** file, in Windows PowerShell. The dataset comprises 1 million sentences randomly sampled from English Wikipedia. While running the code, ensuring that the data folder has ***wiki1m_for_simcse.txt*** file is required.
### Training
- To train the unsupervised contrastive learning approach, run `main.py`.
- All hyperparamters to control model training and the paths to input and output data directories are provided in the given `main.py` file. The values of these hyperparameters can be altered to see how the approach performs in different hyperparameter settings.
- `model.py` file contains the implemented unsupervised contrastive learning-based language model approach that is being imported to `main.py` file in order to train and test the same approach.
- This unsupervised contrastive learning approach can also be extended to perform text similarity using the preTrained language model, the example of which can be found in `main.py` file.
## Text Similarity
### Input Sentences
```
"chocolates are my favourite items.",
"The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
"The person box was packed with jelly many dozens of months later.",
"white chocolates and dark chocolates are favourites for many people.",
"I love chocolates.",
"Let me help you.",
"There are some who influenced many.",
"Chips are getting more popular these days.",
"There are tools which help us get our work done.",
"Electric vehicles are worth buying given their mileage on the road.",
"NATO is the most powerful military alliance.",
"Gone are the days when people got worry about their diets."
```
### Sentence to be compared with the other given sentences
```
"chocolates are my favourite items."
```
### Similarity Scores
<pre>
0.84871584, 0.8847387, 0.874104, <i><b>0.96159446</b></i>, 0.87748206,
0.88612396, 0.9087229, 0.86401033, 0.90140533, 0.8532164,
0.8539922
</pre>

### Analysis
Results, mentioned above, show higher similarity between 1<sup>st</sup> and 5<sup>th</sup> sentences than any other possible 1<sup>st</sup> sentence combination with the remaining 10 input sentences.
