# NLP_Sarcasm
Hi! I found an interesting dataset on Kaggle: 
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

It is a list of news articles from Huffington Post and The Onion and the question is simple - is the headline sarcastic or not? It's a fun way to practice your NLP skills.
## Dataset information and preprocessing:
If you take a look at the dataset, you'll see that it contains:

**article_link** - link to the site for additional data

**headline** - the headline

**is_sarcastic** - is the headline sarcastic or not? (1,0)

My plan for preprocessing is to remove all [stopwords](https://gist.github.com/sebleier/554280) that don't add much value (this, me, there etc.)  and reduce the number of different words by transforming them into their stems using PorterStemmer.
I also "tag" headlines when they use exclamation/question marks (!/?). I do this by replacing these symbols with words so they wouldn't be cleaned out when preparing for PowerStemmer. I also considered doing the same when some headlines ended with "...", but only some of the non-sarcastic articles used it. So while it could improve the model using only this specific dataset, it would create false negatives when new sarcastic headlines end with "..."
## Creating a matrix and fitting the model
CountVectorizer was used to transform the list of strings into a matrix of tokens. The max_features argument determines how many most used words to use in the matrix. Then after splitting the dataset into training and test, the model was fit using Naive Bayes. For max_features a range of 100 to 3000 words was tried out and the optimal number turned out to be 1300.
![Finding optimal max words](https://github.com/henryginter/NLP_Sarcasm/blob/master/graph.png)
## Conclusion
The error rate I achieved was 0.29 by using 1300 top used words. The approach is very simple and can definitely be improved upon. In the dataset link you can find an article about this where a hybrid of LSTM and CNN achieved an error rate of only 0.10. 

A lot of the articles use quotation marks for sarcastic and non-sarcastic purposes. If the dataset was a lot bigger I would consider classifying quoted words separately as independent from the same non-quoted words and see how it would affect the outcome.
