#Data Preprocessing:-

# Import the libraries:
import re
from datetime import datetime
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# defining function for cleaning the data:
def preprocess(data):        
   
    pattern='[0-9]+/[0-9]+/\d\d,\s[0-9]+:\d\d\s[a-zA-Z][a-zA-Z]'         # Regex pattern= ' 8/18/21, 9:48 PM '
    messages=re.split(pattern, data)[1:]           #extracting messages from data  #splits the data
    dates=re.findall(pattern,data)                  #extracting dates from data   #findall: Returns a list containing all matches

    df=pd.DataFrame({'user_message':messages, 'message_date':dates})        #merging messages and data into pandas Dataframe
    df['message_date']=pd.to_datetime(df['message_date'],infer_datetime_format=True)   #convert date of string object into python date in format data of 12 hrs
    df.rename(columns={'message_date':'date'}, inplace=True)                #renaming the column


#Seperating user's name and user's messages:
    users = []
    messages = []
    patterns='([\w\W]+):'           #'Subodh Varane:'  as well as '9594918020'
    for message in df ['user_message']:
        entry = re.split(patterns, message)
        if entry[1:]:  # username
            users.append(entry[1])
            messages.append("  ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

#Extracting meta data:
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month_name()
    df['day']=df['date'].dt.day
    df['hour']=df['date'].dt.hour
    df['minute']=df['date'].dt.minute
    df['month_num'] = df['date'].dt.month
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()

    period = []

    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

     

# Sentiment analysis:
    sentiments = SentimentIntensityAnalyzer()
    df['sentiment_scores']= df['message'].apply(lambda message:sentiments.polarity_scores(message))
    df['compound'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
    df['sentiment_type']=''
    df.loc[df.compound>0,'sentiment_type']='Positive'
    df.loc[df.compound==0,'sentiment_type']='Neutral'
    df.loc[df.compound<0,'sentiment_type']='Negative'
           
    return  df




