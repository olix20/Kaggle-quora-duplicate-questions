
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

I spent the first few weeks experimenting with neural networks, motivated by the idea that the same question can be asked in a 'million' ways and simple hand-engineerd features won't generalize well. I tried LSTM, GRU, 1d convolutional networks, attention models, and a combination of them. 

These were all based on Stanford Glove word embeddings. 

I didn't get good results and i gave up in favour of handmade feature engineering at the end. Even more disappointing is that ensumbling/stacking these semantic models with my best xgboost model didn't result in any improvement. 





