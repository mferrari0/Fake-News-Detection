# Fake-News-Detector

In this notebook I train 4 different NLP models for the detection of fake news from headlines:
- Bag of Words + Logistic Regression
- FastText Embeddings + CNN
- ALBERT (pre-trained model from the Tensorflow Hub)
- BERT (pre-trained model from Huggingface)



## The dataset
The dataset, containing news headlines, is collected from two news websites:

- TheOnion: sarcastic versions of current events.
- HuffPost: real (and non-sarcastic) news headlines.

Each record consists of three attributes: 
- is_sarcastic: 1 if the record is sarcastic (0 otherwise) 
- headline: the headline of the news article 
- article_link: link to the original news article. Useful in collecting supplementary data

<img src="https://github.com/mferrari0/Fake-News-Detector/blob/main/Headlines%20example.PNG" width="600" height="300">


## Technologies and Methods  

<img src="https://i.imgur.com/cllEHm0.png" width="1000" height="300">


- NLP
- Python
- Keras
- Tensorflow
- FastText Embedding
- Transformers
- Bert
- Huggingface
- Pandas
- Google Colab





## Approaches
At first I loaded and prepocessed the data. I also divided the data into train and test set. 

Then I tried 4 different models:

### Bag of Words + Logistic Regression 

Bag of Words (BOW) is an algorithm that transforms the text into fixed-length vectors. The algorithm counts the number of times the word is present in a document or, in this case, headline. The word occurrences allow to compare different headlines and evaluate their similarities. BOW represents the sentence as a bag of terms. It doesn’t take into account the order and the structure of the words, but it only checks if the words appear in the document.

Once I have the BOW vector for each headline, I use **Logistic Regression** to train and test the model for detecting fake news headines. Accuracy is around **83%**, which is impressive already considering that BOW doesn't consider the order and structure of the words.

### FastText Embeddings + CNN 

<img src="https://i.imgur.com/6Pk3Nrv.png" width="600" height="300">


Fasttext embedding is a word to vector model: it represents each word as a vector. I used a pretrained model to generate, for each headline, a feature matrix, that is then used as input to a CNN model as shown in the picture. Accuracy is higher than with the previous model, but still not great.

### BERT pre-trained model from the Tensorflow Hub 

<img src="https://skimai.com/wp-content/uploads/2020/03/Screen-Shot-2020-04-13-at-5.59.33-PM.png" width="700" height="300">

BERT is a family of masked-language models published in 2018 by researchers at Google. It has become a ubiquitous baseline in NLP experiments counting over 150 research publications analyzing and improving the model. I downloaded the pre-trained model from the Tensorflow Hub and fine-tuned it with my dataset.
Accuracy increases to **89%**.
Additional details on the model and its use can be found here: https://www.tensorflow.org/text/tutorials/classify_text_with_bert

### BERT pre-trained model from Huggingface

<img src="https://uptime-storage.s3.amazonaws.com/logos/d32f5c39b694f3e64d29fc2c9b988cdd.png" width="200" height="200">

I used the pre-trained model from the Huggingface repository. Accuracy increases to **92%**.
Additional details on the model and its use can be found here:  https://huggingface.co/course/chapter3/1?fw=tf

## SUMMARY

- Bag of Words + Logistic Regression show pretty good results for a quick deployment (83%)

- FastText Embeddings + CNN is also quick method and increases the accuracy to 

- Bert from HuggingFace has the best accuracy with 92% 


### Further steps: 
  - try other transformer-based models available from either the Tensorflow Hub or Huggingface
  - try finetuning GPT-3
