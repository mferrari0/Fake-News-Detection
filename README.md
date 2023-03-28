# Fake-News-Detector
Fake News Detector using headlines. I used different strategies (BOW, FastText Embedding, Transformers) and compared the performance.

In this notebook I train 4 different NLP models for the detection of fake news headlines:
- Bag of Words + Logistic Regression
- FastText Embeddings + CNN
- ALBERT (using the Tensorflow Hub)
- BERT (using the Huggingface repository)

<img src="https://i.imgur.com/cllEHm0.png" width="900" height="600">


### The dataset
The dataset, containing news headlines, is collected from two news websites:

- TheOnion: sarcastic versions of current events.
- HuffPost: real (and non-sarcastic) news headlines.

Each record consists of three attributes: 
- is_sarcastic: 1 if the record is sarcastic (0 otherwise) 
- headline: the headline of the news article 
- article_link: link to the original news article. Useful in collecting supplementary data


### Technologies and Methods  


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



<img src="https://raw.githubusercontent.com/mferrari0/Brain-Tumor-Detection/main/dataset.PNG" width="900" height="600">




### Approaches
At first I loaded and prepocessed the data. I also divided the data into train and test set. 

Then I tried 4 different models:

** Bag of Words + Logistic Regression **

Bag of Words (BOW) is an algorithm that transforms the text into fixed-length vectors. The algorithm counts the number of times the word is present in a document or, in this case, headline. The word occurrences allow to compare different headlines and evaluate their similarities. BOW represents the sentence as a bag of terms. It doesn’t take into account the order and the structure of the words, but it only checks if the words appear in the document.

Once I have the BOW vector for each headline, I use Logistic Regression to train and test the model for detecting fake news headines. Accuracy is around 83%, which is impressive already considering that BOW doesn't consider the order and structure of the words.

** FastText Embeddings + CNN **

<img src="https://i.imgur.com/6Pk3Nrv.png" width="900" height="600">


Fasttext embedding is a word to vector model: it represents each word as a vector. I used a pretrained model to generate, for each headline, a feature matrix, that is then used as input to a CNN model as shown in the picture. Accuracy is higher than with the previous model, but still not great.

** BERT (Bi-directional Encoder Representations from Transformers) from Tensorflow Hub **

<img src="https://nlp.gluon.ai/_images/bert-sentence-pair.png" width="900" height="600">

BERT is a family of masked-language models published in 2018 by researchers at Google. It has become a ubiquitous baseline in NLP experiments counting over 150 research publications analyzing and improving the model. I download the pre-trained model from the Tensorflow Hub and followed the example at this link: https://www.tensorflow.org/text/tutorials/classify_text_with_bert. Accuracy increases to 89%.

** BERT (Bi-directional Encoder Representations from Transformers) from Huggingface  **

<img src="https://uptime-storage.s3.amazonaws.com/logos/d32f5c39b694f3e64d29fc2c9b988cdd.png" width="900" height="600">

I used the pre-trained model from the Huggingface repository and followed the example at this link: https://huggingface.co/course/chapter3/1?fw=tf. Accuracy increases to 92%.

## SUMMARY

- Bag of Words + Logistic Regression show pretty good results for a quick deployment (83%)

- FastText Embeddings + CNN increases the accuracy to 

- Bert from HuggingFace has the best accuracy with 92% 


### Further steps: 
  - try other transformer-based models available from either the Tensorflow Hub or Huggingface
  - try finetuning GPT-3
