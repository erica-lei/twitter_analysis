
# Twitter Sentiment Analysis

### Summary

Using tweepy API as well as vaderSentiment, I extracted 100 of the most recent tweets of various news organizations and performed a sentiment analysis of each tweet, composing of postive, negative, and neutral sentiments. 

With this information, I created a scatter plot that visualized the tweets that were sent out. Each plot point in the scatter diagram represents the compound sentiment of a tweet. I have also created a bar plot that shows the overall aggregated seniments of the last 100 tweets for each news organization. The news organizations I extracted tweets from included BBC, CBS, CNN, Fox and New York Times.



### Results and Analysis

Based on the data collected, I was able to find that many of the news organization has neutral sentiment. However, other than the neutral sentiment, I also found trends regarding tweet polarity of news organizations:

1. Based on the last 100 tweets extracted from new organizations, after averaging the total overall sentiment, FoxTV has the highest positive sentiment while CNN shows the highest negative sentiment in their Tweets. After running this multiple times, FoxTV has greatest consistency for positive sentiment.

2. Based on the scatter diagram that consists of 100 most recent tweets for each news organization, FoxTV has the least negative sentiment. 

3. The scatter diagram shows that NY Times seems to have the least volatility in their sentiment analysis. Most of the tweet polarity is located at -0.15 and at 0.47. 


Limitations of the data may include,the small sample size and as a result, the sentiment analysis will constantly change. Furthermore, variables such as the media's differing internal structures, as well as employees who manage the twitter page differ across news organizations. News anchors may present their information differently than those whom manage and post the news content on twitter. 


```python
#import dependencies

import pandas as pd
import requests
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from pprint import pprint
import numpy as np

from datetime import datetime

from localenv import access_token , access_token_secret, \
                        consumer_key , consumer_secret
    

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,parser=tweepy.parsers.JSONParser())
```


```python
# BBC, CBS, CNN, Fox, New York times

target_users=("@BBC", "@CBS", "@CNN", "@FoxTV", "@NYTimes")

#array to hold sentiments
sentiment_array=[]
compound_overall=[]


#create a for loop for each news channel
for target in target_users:
    counter=0
    oldest_tweet=None

    
    #make two requests of 50 tweets each
    for x in range(2):
        public_tweets=api.search(
            target,count=50,result_type="recent",max_id=oldest_tweet)    
        for tweet in public_tweets["statuses"]:
            counter = counter + 1
            results = analyzer.polarity_scores(tweet['text'])
            compound=results["compound"]
            positive=results["pos"]
            negative=results["neg"]
            neutral=results["neu"]
            rawtime=tweet["created_at"]
            converted_time=datetime.strptime(rawtime,"%a %b %d %H:%M:%S %z %Y")
            
            
            sentiment_array.append({
                "Users":target,
                "Compound List": compound,
                "Number Tweets Ago":counter,
                "Positive List": positive,
                "Negative List": negative,
                "Neutral List": neutral,
                "Tweet Time":converted_time
            })
            
            
        compound_mean=np.mean(compound)
            
        compound_overall.append({
            "Users":target,
            "Overall Sentiment":compound_mean})
        

        oldest_tweet=int(tweet["id_str"]) -1
     
    
    
```


```python
sentiment_df=pd.DataFrame.from_dict(sentiment_array)
sentiment_df.head()

sentiment_df.to_csv("tweet_sentiment_csv",index=False)
```


```python
overall_sentiment_df = pd.DataFrame(compound_overall)
total_overall_senti= overall_sentiment_df.groupby("Users")["Overall Sentiment"].mean().reset_index()

total_overall_senti.to_csv("overall_sentiment_csv",index=False)
total_overall_senti


```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Users</th>
      <th>Overall Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBS</td>
      <td>-0.11315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>-0.26335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxTV</td>
      <td>0.27870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@NYTimes</td>
      <td>-0.07655</td>
    </tr>
  </tbody>
</table>
</div>




```python
bbc_senti = total_overall_senti["Overall Sentiment"][0]
cbs_senti = total_overall_senti["Overall Sentiment"][1]
cnn_senti = total_overall_senti["Overall Sentiment"][2]
fox_senti = total_overall_senti["Overall Sentiment"][3]
nyt_senti = total_overall_senti["Overall Sentiment"][4]
```


```python
sns.lmplot(x="Number Tweets Ago", y="Compound List", data=sentiment_df, 
           fit_reg=False, hue="Users", palette="husl",
           legend=True,
           scatter_kws={
               "edgecolor":"grey",
               "linewidth":0.5,
               "alpha":0.7
           })

plt.title("Sentiment Analysis of Media Tweets \n 03 / 04 / 2018", size=16)
plt.ylabel("Tweet Polarity", size=13)
plt.xlabel("Tweets Ago", size=13)

plt.savefig("sentimentfig")
```


![png](output_7_0.png)



```python
y_axis = total_overall_senti["Overall Sentiment"] 
x_labels = total_overall_senti["Users"]


plt.figure(figsize=(9, 6))

ax = y_axis.plot(kind='bar')
ax.set_xticklabels(x_labels)
ax.axhline(y=0, linestyle="dashed", color="black",linewidth=0.5)

rects = ax.patches

rects[0].set_color("red")
rects[1].set_color("gold")
rects[2].set_color("teal")
rects[3].set_color("skyblue")
rects[4].set_color("purple")



ax.set_title("Overall Media Sentiment on Twitter \n 03 / 04 / 2018", size = 16)

plt.xlabel("Users", size = 14)
plt.xticks(rotation=0)
plt.ylabel("Tweet Polarity", size = 14)
plt.ylim((min(y_axis) - .2) , (max(y_axis)+ .2))
plt.grid(linestyle="dashed",color="skyblue",alpha=0.3)


# For each bar: Place a label
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 5
    va = 'bottom'
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'
    label = "{:.3f}".format(y_value)

    # Create annotation
    plt.annotate(
        label,                      
        (x_value, y_value),        
        xytext=(0, space),         
        textcoords="offset points",
        ha='center',               
        va=va)                      
                                    
        
plt.savefig("overallsentiment")
```


![png](output_8_0.png)

