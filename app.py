#Deployment:-

#Importing libraries:  
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import matplotlib.image as img
import squarify
import seaborn as sns

#Uploading header files:
st.title("WhatsApp Chat Sentiment Analyzer")
st.image("WhatsAppimg.jpg", width=400)
st.markdown("## Hey what's up? This is streamlit app which analyzes user's individual as well as group chats and displays it in form of visualizations. ")
"## To upload a file go through some steps below : "
st.text("1. Open the individual or group chat.")
st.text("2. Tap More options > More > Export chat.")
st.text("3. Choose whether to export with media or without media.")
st.image("whatsapp_exportchat.jpg",  width=500)


#Uploading files into sidebar:
st.sidebar.title("Welcome !")
uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()           
    data = bytes_data.decode("utf-8")       #convert data stream (bytes) to data string
    df= preprocessor.preprocess(data)       #func for prerprocessing data
    st.dataframe(df)                        #display dataframe in streamlit

#Displaying user names in userlist of groups :
    user_list = df['user'].unique().tolist()# fetch unique users:
    user_list.remove('group_notification')
    user_list.sort()                        #sorting userlist
    user_list.insert(0,"Overall")           #group analysis at 0 position 

    selected_user=st.sidebar.selectbox("Show analysis wrt",user_list)

# Implemening stats :

# frontend: 
    if st.sidebar.button("Show Analysis"):

        num_messages, num_words, num_letters, num_media , num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics :")
        col1, col2, col3, col4 ,col5 = st.columns(5)
        with col1:
            st.header("Total message")          #total messages
            st.title(num_messages)
        with col2:
            st.header("Total words")             #total words
            st.title(num_words)
        with col3:
            st.header("Total letters")             #total letters
            st.title(num_letters)
        with col4:
            st.header("Media Shared")             #total Media shared
            st.title(num_media)
        with col5:
            st.header("Links Shared")              #total link shared
            st.title(num_links) 
            


 # finding the busiest users in the group(Group Level):
        if selected_user == 'Overall':
            st.title('Most Active Users :')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                st.title("(Bar Plot)")
                ax.bar(x.index, x.values, color = 'orange', edgecolor= "black", alpha=0.8)
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)

            with col2:
                st.title("(List of Users)")
                st.dataframe(new_df)    


# Activity map(bar graph):
        
        col1, col2 = st.columns(2)
 
        with col1:
            st.header("Most Active Day :")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red', edgecolor="black")
            plt.xticks(rotation='horizontal')
            st.pyplot(fig)

        with col2:
            st.header("Most Active Month :")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='blue',edgecolor="black")
            plt.xticks(rotation='horizontal')
            st.pyplot(fig)


# most common words:
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1], color= 'grey')
        plt.xticks(rotation='vertical')
        st.title('Most Common Words :')
        st.pyplot(fig)


# WordCloud:
        st.title("WordCloud ")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        plt.axis("off")
        ax.imshow(df_wc)
        st.pyplot(fig)



# Emoji analysis:
        st.title("Emoji Analysis :")
        emoji_df = helper.emoji_helper(selected_user,df)
        
        col1,col2 = st.columns(2)

        with col1:
            st.title("(No. of emojis)")
            st.dataframe(emoji_df)
        with col2:
            st.title("(Pie-Chart)")
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.3f")
            st.pyplot(fig)




# User activity map:
# timeline graph:

        st.title("User Activity Map ")
        if selected_user == 'Overall':
            st.title("Monthly Timeline :")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            st.title("Daily Timeline :")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='brown')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


#Weekly activity map(heatmap):

        st.title("Weekly Activity Map ")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        plt.title("2D HeatMap",fontsize=23,fontweight="bold")
        st.pyplot(fig)       



# Sentimental Analysis(treemap):

        st.title("Sentiment Analysis :")
        sentiment_df = helper.sentiment_analysis(selected_user, df)
        col1,col2 = st.columns(2)

        with col1:
            st.title("(Sentiment Scores)")
            st.dataframe(sentiment_df)
        with col2:
            st.title("(TreeMap)")
            fig, ax= plt.subplots()
            squarify.plot(sizes=sentiment_df.values, label=["NEUTRAL","POSITIVE", "NEGATIVE"],pad=3,color=["blue","green","red"], alpha=0.5,edgecolor="black",text_kwargs={'fontsize':15})
            fig.set_size_inches(15, 10.5)
            st.pyplot(fig)
                         

#End Plot(Image):

        col1, col2 = st.columns(2)

        with col1 :
            st.title("Thank You :)") 

        with col2:
            st.title("-By Subodh Ajay Varane")
            testImage = img.imread('mypic.png')
            plt.axis('off')
            plt.imshow(testImage)
            st.image(testImage)




