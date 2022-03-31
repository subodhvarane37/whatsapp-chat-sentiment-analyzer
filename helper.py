#Exploratory data analysis (EDA):-

#Importing libraries :
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import emoji
import squarify


extract= URLExtract()

def fetch_stats(selected_user, df):

    if selected_user !='Overall':
        df=df[df['user']== selected_user]

# total number of messages:
    num_messages=df.shape[0]

#  total number of words:
    words=[]
    for message in df ['message']:
        words.extend(message.split())
   
#total number of letters :
    df['letters_count'] = df['message'].apply(lambda s : len(s))
    num_letters = np.sum(df.letters_count)

# total number of media :
    media = r'<Media omitted>'
    df['media_count'] = df.message.apply(lambda x : re.findall(media, x)).str.len()
    num_media = np.sum(df.media_count)

# number of links shared :
    links=[]
    for message in df ['message']:
        links.extend(extract.find_urls(message))
    
    return num_messages, len(words), num_letters, num_media , len(links)


# most busy users (bar graph) :

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df


#most busy day:

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

# most busy month :

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

# most common words :

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


#Wordcloud :

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


#emoji analysis:

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


#monthly timeline:

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline



#heatmap activity :

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return activity_heatmap


# Sentiment Analysis:

def sentiment_analysis(selected_user, df):
    if selected_user != "Overall":
        df= df[df['user'] == selected_user]
     
    sentiment_df=df['sentiment_type'].value_counts().to_frame()
    return sentiment_df
     




