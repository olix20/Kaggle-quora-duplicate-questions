
## Kaggle-Quora duplicate question pairs competition 

"Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this compettiion, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers."

https://www.kaggle.com/c/quora-question-pairs/


## My solution 

### preprocessing
I did a few rounds of data cleaning. version A is more conservative by removing pretty much all non alphanumerical characters, while version B keeps more of the raw features. 

You can find the process in text_cleaning.ipynb


### semantic matching

~/source/semantic models/

I spent the first few weeks experimenting with neural networks, motivated by the idea that the same question can be asked in a 'million' ways and simple hand-engineerd features won't generalize well. I tried LSTM, groups, 1d convolutional networks, attention models, and a combination of them. 

These were all based on Stanford Glove word embeddings. 

I didn't get good results and i gave up in favour of handmade feature engineering at the end. Even more disappointing is that ensumbling/stacking these semantic models with my best xgboost model didn't result in any improvement. 

After competition ended, someone posted a slightly different solution than mine claiming to have superb results:
https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23

the difference is in concatenating encodings of q1 and q2, versus, using their diffs and multiplies, which may avoid overfitting to training data which were quite different than test data. 


### Feature engineering

~/source/feature engineering/

A combination of NLP and graph-based features. 

NLP: letter n-grams (1,4), word n-grams(1,3), differences of them, SVD and NMF of their vertical/horizontal concatenations

Graph-based (aka leaks): so basically you create a graph from a concatination of training and test data, with nodes being questions and edges if q1 and q2 appear in training or test set. Most notably, number of common neighbours has a significant effect on results. other features i created include page rank, number of triangles crossing the node, count of their common second degree neighbours, max k-core of q1 and q2. 

I suspect in real world situations graph features will be of any use though. 


### Models 

My dominant model and final submission was a single XGBoost model with 200+ features. Public leaderboard score was 0.14932 and private LB score 0.14634.

I'm pretty sure towards the end everyone gained significantly from stacking/bagging while i didn't. I tried a few bagging and 1/2 level stacking approache (based on out of fold predictions of individual models) with no gains. I might have done something wrong in the process or a bug might have been there; whatever it is, while being disappointing it's a learning lesson to invest more time in a solid stacking pipeline early on. 



