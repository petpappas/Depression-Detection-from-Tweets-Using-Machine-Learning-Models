# Generated from: Final Hackathon.ipynb
# Converted at: 2026-03-02T20:57:16.837Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## 1️⃣ Data Understanding & Cleaning


# Importing the necessary libraries for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset from a CSV file into a pandas DataFrame
Final=pd.read_csv("clean_tweet_Dec19ToDec20.csv")

# Displaying a message to indicate that basic dataset information follows
print("Some basic information about this dataset:")

# Printing structural information about the dataset (columns, non-null counts, data types)
print(Final.info())

#Now,lets set Unnamed: 0 as row names
# Setting the column "Unnamed: 0" as the index of the DataFrame
Final=Final.set_index("Unnamed: 0")

#Now,lets delete row zero
# Dropping the row with index value 0 (likely an unnecessary or corrupted entry)
Final=Final.drop(0)

# Printing a blank line for better output readability
print("\n")

# Displaying the updated DataFrame after index setting and row deletion
print(Final)

#Now,lets check how many zeros and ones we have.Lets express these numbers in percentages
# Getting the total number of rows and columns in the dataset
rows,columns=Final.shape

# Printing a blank line for formatting
print("\n")

# Printing the number of rows
print("Rows equal to:", rows)

# Printing the number of columns
print("Columns equal to:", columns)

#Now,lets check if there are any missing values on the sentiment column of this dataset 
# Printing a blank line for formatting
print("\n")

# Counting and printing the number of missing (NaN) values in the sentiment column
print("The NaN values of sentiment equal to:",Final["sentiment"].isnull().sum()) #So,there are no missing values in the sentiment

# Counting and printing the number of missing (NaN) values in the text column
print("The final NaN values of text equal to:",Final["text"].isnull().sum()) 

#Now,lets check what percentage counts for values zero on the dataset and what percent counts for values one.
# Printing a blank line for formatting
print("\n")

# Calculating the percentage of rows where sentiment equals 1
percentage_1=((Final["sentiment"].value_counts()[1]/rows)*100).round(2)
print("Percentage of 1 equal to:",percentage_1 , "%")

# Calculating the percentage of rows where sentiment equals 0
percentage_2=((Final["sentiment"].value_counts()[0]/rows)*100).round(2)
print("Percentage of 1 equal to:",percentage_2 ,"%")

# Printing a blank line for formatting
print("\n")

#So,we can say that the sample is pretty well separated.

#Firstly,lets check the NaN text values.
# Displaying rows where the text column contains NaN values
Final[Final["text"].isna()] #We see that all of these rows are assigned the zero index.

#Next step is actually to delete all these rows because actually they offer us no information.
# Removing rows where the text column has missing values
Final.dropna(subset="text", inplace=True) #Now,all the NaN rows are deleted.


#Now,we need to define what values 0 and 1 indicate
# Creating a list of words that are typically associated with positive sentiment
positive_words=["happy","love","loving","joy","glad","pride","amazing","wonderful","perfect","awesome","love","great","good","nice","lovely","kind","beautiful","honest"]

# Looping through each positive word to check how often it appears
for x in positive_words:
 print(f"{x}")
 # Filtering rows where the text contains the specific positive word (case-insensitive)
 # Then counting how many times each sentiment value appears in those filtered rows
 print((Final[Final['text'].str.contains(x, case=False, na=False)]["sentiment"].value_counts()))
 print("\n")

# Creating a list of words that are typically associated with negative sentiment
negative_words=["sad","sadness","lonely","alone","hurt","depressed","depression","anxiety","hopeless","crying","anger","angry","hate","mad","upset","awful","horrible","terrible","broken","pain","tired","suicide","worst","sick","evil"]

# Looping through each negative word to check how often it appears
for y in negative_words:
 print(f"{y}")
 # Filtering rows where the text contains the specific negative word (case-insensitive)
 # Then counting how many times each sentiment value appears in those filtered rows
 print((Final[Final['text'].str.contains(y, case=False, na=False)]["sentiment"].value_counts()))
 print("\n") 

#It is also a really good idea to check if there are any duplicates in this specific dataset 
# Printing a blank line for formatting
print("\n")

# Counting duplicated values in the text column
duplicates=Final["text"].duplicated().sum()
print("Duplicates in this dataset equal to:",duplicates) #Actually, there are 10.314 duplicates in this specific dataset.

# Printing a blank line for formatting
print("\n")

#So now we need to drop the duplicates
# Removing duplicate rows based on the text column to avoid repeated information
Final=Final.drop_duplicates(subset=['text'])

print("After removing the duplicates we count them again to be sure")

# Re-checking the number of duplicated text values to confirm removal
print("After counting them again they equal to:",Final["text"].duplicated().sum())#Here give an explanation why did you drop the duplicates.

# Since duplicated text entries do not provide additional information 
# and may bias the analysis or model performance, we remove them 
# to ensure data quality and prevent over-representation of repeated tweets.

#Since, we end up in a conclusion that they offer not important information to us we delete them.
# Keeping only rows where the text contains more than one word 
# (removing extremely short entries that may not carry meaningful information)
Final=Final[Final["text"].str.split().str.len()>1]

#Now,lets also apply a Pie-Chart to indicate the percentages of each sentiment(0-1).

# Importing matplotlib for visualization
import matplotlib.pyplot as plt 

# Creating a new figure with a specific size for better readability
plt.figure(figsize=(10,8))

# Creating a pie chart based on the sentiment value counts
# sort_index() ensures that sentiment 0 and 1 appear in correct numerical order
# startangle rotates the chart for better visual alignment
# autopct formats the percentage labels inside the pie
# colors assigns specific colors to each sentiment category
# labels defines the meaning of each class (0 and 1)
plt.pie(Final['sentiment'].value_counts().sort_index(),
        startangle=140,
        autopct='%1.1f%%',
        colors=["red","green"],
        labels=['Depression (0)', 'Not Depression (1)'])

# Adding a title to clearly describe what the chart represents
plt.title("Depression(0) vs No Depression(1)")

# Displaying the final pie chart
plt.show()

# ## 2️⃣ Text Preprocessing


#Now,lets make sure  that any URL,mentions and Tags are  removed from this dataset.

# Importing the regular expressions library for pattern-based text cleaning
import re

# Defining a function that cleans tweet text using regex substitutions
def clean_tweet(text):
    
    # 1. Remove URLs (http or https)
    # This pattern removes any substring that starts with http or https followed by non-space characters
    text = re.sub(r'https?://\S+', '', text)
    
    # 2. Remove Mentions (@username)
    # This removes Twitter mentions that start with @ followed by word characters
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove Hashtag symbol (#) but keep the text
    # This removes only the '#' character while preserving the actual hashtag word
    text = re.sub(r'#', '', text)
    
    # 4. Remove RT (Retweet) tag
    # This removes the 'RT' indicator typically found in retweets
    text = re.sub(r'RT\s+', '', text)
    
    # 5. Remove extra spaces
    # This replaces multiple whitespace characters with a single space
    # and removes leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Returning the cleaned text
    return text

# Apply the function to the column
# This applies the clean_tweet function to every row in the 'text' column
Final['text'] = Final['text'].apply(clean_tweet)

# Displaying the first 10 cleaned text entries to verify the transformation
print(Final['text'].head(10))

# Importing necessary NLP libraries from NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

# 1. Required downloads (including omw-1.4 for extended WordNet support)
# Downloading stopwords corpus for removing common words
nltk.download('stopwords')

# Downloading tokenizer models
nltk.download('punkt')

# Downloading POS tagger model
nltk.download('averaged_perceptron_tagger')

# Downloading WordNet lexical database
nltk.download('wordnet')

# Downloading Open Multilingual WordNet (required for proper lemmatization support)
nltk.download('omw-1.4')  # <--- Added for extended lemma support!

# 2. Initializing NLP tools
# Creating a WordNet lemmatizer instance
lemmatizer = WordNetLemmatizer()

# Creating a set of English stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# 3. POS mapping function
# This function maps NLTK POS tags to WordNet POS tags
# WordNet requires specific POS format for accurate lemmatization
def get_wordnet_pos(treebank_tag):
    
    # Adjectives
    if treebank_tag.startswith('J'):
        return wn.ADJ
    
    # Verbs
    elif treebank_tag.startswith('V'):
        return wn.VERB
    
    # Nouns
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    
    # Adverbs
    elif treebank_tag.startswith('R'):
        return wn.ADV
    
    # Default to noun if no match is found
    else:
        return wn.NOUN

# 4. Unified loop with Lemmatization & Stopwords removal
# Creating an empty list to store processed tweets
lemmatized_tweets = []

# Iterating over each sentence in the dataset
for sentence in Final["text"]:
    
    # Tokenization and lowercase conversion
    # Converting sentence to string (for safety) and lowercasing it
    tokens = word_tokenize(str(sentence).lower())
    
    # Applying Part-of-Speech tagging to each token
    tagged = pos_tag(tokens)
    
    # Creating a cleaned and lemmatized version of the sentence
    words_row = [
        # Lemmatizing each word using its mapped POS tag
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
        
        # Iterating over each word-tag pair
        for word, tag in tagged 
        
        # Keeping only alphabetic words (removes numbers & symbols)
        if word.isalpha()
        
        # Removing stopwords
        and word not in stop_words
        
        # Removing very short words (length <= 2)
        and len(word) > 2
    ]
    
    # Joining processed words back into a cleaned sentence
    lemmatized_tweets.append(" ".join(words_row))

# Saving the processed text into a new DataFrame column
Final["text_lemmatized"] = lemmatized_tweets

# Printing confirmation message
print("✔ Lemmatization & Stopwords removal completed with OMW 1.4!")

# Displaying original and processed text side by side for comparison
print(Final[["text", "text_lemmatized"]].head())

#Now,lets work on the differences between the different tweets categories.
Final['word_count'] = Final['text_lemmatized'].apply(lambda x: len(str(x).split()))
#Now,lets check on the mean value for each category
print("Μean number of words per different class:")
print(Final.groupby(["sentiment"])["word_count"].mean())
print("\n")
plt.pie(Final.groupby(["sentiment"])["word_count"].mean(),startangle=140,autopct='%1.1f%%',colors=["orange","yellow"],labels=['Depression (0)', 'Not Depression (1)'])
plt.title("Depression(0) Word Count mean()  vs No Depression(1) Word Count mean()")
plt.show()
#So, as we can conclude, there is no significant statistical difference between mean values for zero and one.

#Now,lets take a look at the character count for each different condition.
Final['char_count'] = Final['text_lemmatized'].str.len()
print("\n")
print("Number of character per class:")
print(Final.groupby(["sentiment"])["char_count"].mean())
print("\n")
plt.pie(Final.groupby(["sentiment"])["char_count"].mean(),startangle=140,autopct='%1.1f%%',colors=["grey","red"],labels=['Depression (0)', 'Not Depression (1)'])
plt.title("Depression(0) Word Count mean()  vs No Depression(1) Word Count mean()")
plt.show()

#  (KDE Plot)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(data=Final, x='char_count', hue='sentiment', fill=True, palette=['red', 'green'])
plt.title('Distribution of Character Count')

plt.subplot(1, 2, 2)
sns.kdeplot(data=Final, x='word_count', hue='sentiment', fill=True, palette=['red', 'green'])
plt.title('Distribution of Word Count')

plt.tight_layout()
plt.show()

# The average number of words and characters per tweet is very similar across both sentiment classes. Tweets in class 1 are only slightly longer than those in class 0, with a difference of less than one word and only a few characters on average. This minimal variation suggests that text length is not a meaningful discriminative feature in the dataset. Therefore, the differences in model performance are more likely driven by lexical content and contextual patterns rather than structural differences in tweet length.


def get_sentiment_split(word_list, df):
    # This function calculates how many times each word 
    # appears in tweets with sentiment 0 and sentiment 1
    # From the list of positive and negative words we set at the beginning.
    data = [] # List that will store the results
    for word in word_list:
        word = word.lower() # Convert word to lowercase for consistent matching
        # Filter the dataframe to find tweets that contain the word
        # We use the lemmatized text for more accurate token matching
        mask = df["text_lemmatized"].apply(lambda x: word in str(x).split())
        subset = df[mask]
        counts = subset['sentiment'].value_counts()
        
         # Add frequency for sentiment class 0
        data.append({'Word': word, 'Sentiment': 0, 'Frequency': counts.get(0, 0)})
         # Add frequency for sentiment class 1
        data.append({'Word': word, 'Sentiment': 1, 'Frequency': counts.get(1, 0)})
        
    return pd.DataFrame(data)

# Now we call the function separately for positive and negative word lists
pos_df = get_sentiment_split(positive_words, Final)
neg_df = get_sentiment_split(negative_words, Final)
# 4. Combine both dataframes into one (optional, useful for unified analysis)
combined_df = pd.concat([pos_df, neg_df])

# 4. Combine both dataframes into one (optional, useful for unified analysis)
plt.figure(figsize=(20, 10)) # Large width to fit all words horizontally
sns.barplot(data=pos_df, x='Word', y='Frequency', hue='Sentiment', palette=['red', 'green'])
plt.title("Frequency of Emotional Words by Sentiment", fontsize=16)
plt.xlabel("Words")
plt.ylabel("Number of occurences")
plt.legend(title='Category')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# Create the barplot for negative words (horizontal orientation)
plt.figure(figsize=(20, 25)) 
sns.barplot(data=neg_df, x='Frequency', y='Word', hue='Sentiment', palette=['red', 'green'])
plt.title("Frequency of Emotional Words by Sentiment", fontsize=16)
plt.xlabel("Number of occurences", fontsize=10)
plt.ylabel("Words")
plt.legend(title='Category')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# ## 3️⃣ Exploratory Data Analysis (EDA)


# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Now,lets get the text from 0 and 1

# Filtering the dataset to get only tweets labeled as 0 (Depression)
zero_number=Final[Final["sentiment"]==0]["text_lemmatized"]

# Filtering the dataset to get only tweets labeled as 1 (Not Depression)
one_number=Final[Final["sentiment"]==1]["text_lemmatized"]

# Initializing the TF-IDF Vectorizer
# stop_words='english' removes common English stopwords
# max_features=30 limits the vocabulary to the top 30 features based on TF-IDF
vectorizer = TfidfVectorizer(stop_words='english',max_features=30)

# Fitting the vectorizer only on depressive tweets (sentiment = 0)
# This learns the most important words (based on TF-IDF) for class 0
vectorizer.fit(zero_number)

# Extracting the feature names (top words) selected by TF-IDF
top_words_0=vectorizer.get_feature_names_out()

# Printing the selected top words for sentiment 0
# Note: The results are returned in alphabetical order, not in order of importance.
print("The most common words for sentiment 0 based on alphabetical order are:")
print(top_words_0)

#Now,lets apply the same thing for one_number

# Fitting the vectorizer on non-depressive tweets (sentiment = 1)
# This recalculates the top 30 important words for class 1
vectorizer.fit(one_number)

# Extracting the feature names (top words) for sentiment 1
top_words_1=vectorizer.get_feature_names_out()

# Printing the selected top words for sentiment 1
print("\n")
print("The most common words for sentiment 1 based on alphabetical order are:")
print(top_words_1)

#Now,we we will define the most important unigrams  for sentiment=1 based on importance.
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 2. Training phase (fit_transform)
vectorizer = TfidfVectorizer(stop_words='english', max_features=30)
tfidf_matrix = vectorizer.fit_transform(one_number)
feature_names = vectorizer.get_feature_names_out()
# Now,we get the mean score value for each word in all tweets.
scores = tfidf_matrix.mean(axis=0).tolist()[0]
#Dataframe creation for all the words and its scores.
importance_df = pd.DataFrame({'word': feature_names, 'score': scores})
importance_df = importance_df.sort_values(by='score', ascending=False)

print(importance_df)

#Now,lets create a bar chart for the one classification.
#Now,lets also create a bar chart to make it a bit clearer
plt.figure(figsize=(14,9))
sns.barplot(x='score',y='word', data=importance_df)
plt.xlabel('Score_count')
plt.ylabel('Words')
plt.title("Bar Chart of Importance for Non Depression Classification")
plt.show()

# Now, let's apply the same procedure for the zero people (Depression / "sad") tweets

# Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. Training the TF-IDF vectorizer on depressive tweets
# stop_words='english' removes common words, max_features=50 limits to top 50 terms
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_matrix = vectorizer.fit_transform(zero_number)

# 3. Getting the feature names (words) for sentiment 0
feature_names_zero = vectorizer.get_feature_names_out()

# 4. Correction: calculating the average TF-IDF score for each word across all tweets
# This gives a better measure of overall importance rather than just alphabetical order
scores_zero = tfidf_matrix.mean(axis=0).tolist()[0]

# 5. Creating a DataFrame with words and their TF-IDF scores
importance_df_zero = pd.DataFrame({'word': feature_names_zero, 'score': scores_zero})

# 6. Sorting the DataFrame from most important to least important word
importance_df_zero = importance_df_zero.sort_values(by='score', ascending=False)

# Displaying the sorted DataFrame
print(importance_df_zero)

# Now, let's create a bar chart for the Depression classification

# Setting figure size for readability
plt.figure(figsize=(14,9))

# Creating a horizontal bar plot showing word importance scores
sns.barplot(x='score', y='word', data=importance_df_zero)

# Labeling axes
plt.xlabel('Score_count')
plt.ylabel('Words')

# Adding a descriptive title
plt.title("Bar Chart of Importance for Depression Classification")

# Displaying the bar chart
plt.show()

#  ## 4️⃣ Feature Engineering (Vectorization)


# Now, let's work a bit on the single words from 0 and 1

# Firstly, we need to check if there are any common words in both positive_words (Not Depression) and negative_words (Depression)
common_words = []

# Iterating through the top words in the non-depressive (positive) importance DataFrame
for word in importance_df['word']:
    
    # Checking if the word also appears in the depressive (negative) importance DataFrame
    if word in importance_df_zero['word'].values:
        common_words.append(word)

# Printing the number of common words between the two top word lists
print(len(common_words))

# Observation:
# Because out of the first 50 most important words for each category, 33 are common,
# we cannot make any safe conclusions about single-word distinctions in this dataset.

# Conclusion:
# To gain better insight, we will continue our analysis using bigrams (two-word combinations),
# as they are more likely to capture contextual differences between depressive and non-depressive tweets.

# Now, let's apply the word counter for these words using WordCloud

# Importing the WordCloud library
# !pip install wordcloud  # Uncomment if WordCloud is not installed
from wordcloud import WordCloud

# Filtering the importance DataFrames to keep only words that are unique (not common) between the two classes
unique_pos = importance_df[~importance_df['word'].isin(common_words)]
unique_neg = importance_df_zero[~importance_df_zero['word'].isin(common_words)]

# ===============================
# WordCloud for Positive Words (Not Depression)
# ===============================

# Creating a dictionary of words and their corresponding TF-IDF scores for positive tweets
word_scores_pos = dict(zip(unique_pos['word'], unique_pos['score']))

# Setting figure size for the WordCloud visualization
plt.figure(figsize=(12,6))

# Generating the WordCloud from the word frequencies
wc_pos = WordCloud(width=800, height=400,
                   background_color='white',
                   colormap='Greens') \
         .generate_from_frequencies(word_scores_pos)

# Displaying the WordCloud image
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')  # Hiding the axes for clarity
plt.title("Unique Positive Words")
plt.show()
print('\n')


# WordCloud for Negative Words (Depression)


# Creating a dictionary of words and their corresponding TF-IDF scores for negative tweets
word_scores_neg = dict(zip(unique_neg['word'], unique_neg['score']))

# Setting figure size for the WordCloud visualization
plt.figure(figsize=(14,6))

# Generating the WordCloud from the word frequencies
wc_neg = WordCloud(width=800, height=400,
                   background_color='white',
                   colormap='Reds') \
         .generate_from_frequencies(word_scores_neg)

# Displaying the WordCloud image
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')  # Hiding the axes for clarity
plt.title("Unique Negative Words")
plt.show()

# Importing Counter to calculate word frequencies efficiently
from collections import Counter

# Separating depressive (0) and non-depressive (1) tweets
depressive_tweets     = Final[Final["sentiment"] == 0]["text_lemmatized"]
not_depressive_tweets = Final[Final["sentiment"] == 1]["text_lemmatized"]

# Function to calculate word frequencies in a series of tweets
def word_freq(series):
    counter = Counter()
    for tweet in series:
        # Splitting each tweet into words and updating the counter
        counter.update(str(tweet).split())
    return counter

# Calculating word frequencies for depressive tweets
freq_dep     = word_freq(depressive_tweets)

# Calculating word frequencies for non-depressive tweets
freq_not_dep = word_freq(not_depressive_tweets)

# Creating a set of all unique words across both classes
all_words = set(freq_dep.keys()) | set(freq_not_dep.keys())

# Calculating the ratio of depressive vs non-depressive occurrences for each word
ratio_data = []
for word in all_words:
    c0 = freq_dep.get(word, 0)       # Frequency in depressive tweets
    c1 = freq_not_dep.get(word, 0)   # Frequency in non-depressive tweets
    total = c0 + c1                  # Total occurrences
    if total == 0:
        continue
    ratio = c0 / total               # Proportion in depressive class
    ratio_data.append({"word": word, "freq_dep": c0, "freq_not_dep": c1,
                        "total": total, "ratio": ratio})

# Converting the ratio data into a DataFrame
ratio_df = pd.DataFrame(ratio_data)

# Thresholds for filtering neutral words
MIN_FREQ   = 50
RATIO_LOW  = 0.35
RATIO_HIGH = 0.65

# Selecting neutral words that appear frequently in both classes
neutral_words_df = ratio_df[
    (ratio_df["ratio"]        >= RATIO_LOW)  &
    (ratio_df["ratio"]        <= RATIO_HIGH) &
    (ratio_df["freq_dep"]     >= MIN_FREQ)   &
    (ratio_df["freq_not_dep"] >= MIN_FREQ)
].sort_values("total", ascending=False)

# Printing the number of neutral words found
print(f"✔ It was found {len(neutral_words_df)} neutral words that will be removed")

# Displaying top 20 neutral words with frequencies and ratio
print(neutral_words_df[["word", "freq_dep", "freq_not_dep", "ratio"]].head(20).to_string(index=False))

# Creating a list of custom stopwords to remove neutral words later
custom_stopwords = list(neutral_words_df["word"])

# Preparing top neutral words for visualization
top_neutral = neutral_words_df.head(30).copy()
top_neutral_melted = top_neutral.melt(
    id_vars="word", value_vars=["freq_dep", "freq_not_dep"],
    var_name="Class", value_name="Frequency"
)
# Mapping column names to readable class labels
top_neutral_melted["Class"] = top_neutral_melted["Class"].map(
    {"freq_dep": "Depressive (0)", "freq_not_dep": "Not Depressive (1)"}
)


# Barplot of Top Neutral Words

plt.figure(figsize=(14, 6))
sns.barplot(data=top_neutral_melted, x="word", y="Frequency", hue="Class",
            palette={"Depressive (0)": "red", "Not Depressive (1)": "green"})
plt.title("Top 20 Neutral Words (appear equally in both classes)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Histogram of Word Ratios

plt.figure(figsize=(10, 5))
# Plotting the distribution of ratios for words with total frequency ≥ 50
sns.histplot(ratio_df[ratio_df["total"] >= 50]["ratio"], bins=30, color="steelblue", kde=True)

# Highlighting the neutral zone in orange
plt.axvspan(RATIO_LOW, RATIO_HIGH, alpha=0.2, color="orange",
            label=f"Neutral zone ({RATIO_LOW}-{RATIO_HIGH})")

# Adding titles and labels
plt.title("Distribution of Word Ratios (total freq ≥ 50)")
plt.xlabel("Ratio (0 = only depressive, 1 = only not depressive)")
plt.legend()
plt.tight_layout()
plt.show()

# Ratio-based neutral word filtering: word frequencies were computed separately for each class. Words with a depressive-class ratio between 0.35 and 0.65 (i.e., appearing roughly equally in both classes) and a minimum absolute frequency of 50 occurrences per class were identified as neutral and added to a custom stopword list. This process yielded 1,656 neutral words, including high-frequency terms such as "india", "get", "like", "time" and "health".


# However, because some words are common in both classes, it is better to also  analyze bigrams (two-word combinations)
# to gain clearer insights for each category.

# Firstly, we apply this for the 0 (Depressed) people

# Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. Training the TF-IDF vectorizer with bigrams
# stop_words='english' removes common words
# max_features=30 limits the vocabulary to the top 30 bigrams
# ngram_range=(2,2) ensures we extract only bigrams (two-word combinations)
vectorizer = TfidfVectorizer(stop_words='english', max_features=30, ngram_range=(2, 2))
tfidf_matrix_double = vectorizer.fit_transform(zero_number)

# 3. Getting the feature names (the bigrams)
feature_names_zero_double = vectorizer.get_feature_names_out()

# 4. Correction: calculating the mean TF-IDF score of each bigram across all depressive tweets
scores_zero_double = tfidf_matrix_double.mean(axis=0).tolist()[0]

# 5. Creating a DataFrame with bigrams and their corresponding TF-IDF scores
importance_df_zero_double = pd.DataFrame({'word': feature_names_zero_double, 'score': scores_zero_double})

# 6. Sorting the DataFrame from most important to least important bigram
importance_df_zero_double = importance_df_zero_double.sort_values(by='score', ascending=False)

# Displaying the sorted DataFrame
print(importance_df_zero_double)

# Creating a bar chart for better visualization of bigram importance
plt.figure(figsize=(12,9))

# Horizontal bar plot showing bigram scores
sns.barplot(x='score', y='word', data=importance_df_zero_double)

# Labeling axes
plt.xlabel('Score_count')
plt.ylabel('Words')

# Adding a descriptive title
plt.title("Bar Chart of Importance for Depression Classification (Bigrams)")

# Displaying the chart
plt.show()

# Now, let's apply the same procedure for Non-Depressed People (1)

# Because some words are common between depressive and non-depressive tweets,
# analyzing bigrams (two-word combinations) helps draw clearer conclusions.

# Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. Training the TF-IDF vectorizer with bigrams for non-depressive tweets
# stop_words='english' removes common words
# max_features=30 limits the vocabulary to the top 30 bigrams
# ngram_range=(2, 2) ensures we extract only bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=30, ngram_range=(2, 2))
tfidf_matrix_double_ones = vectorizer.fit_transform(one_number)

# 3. Getting the feature names (bigrams)
feature_names_ones = vectorizer.get_feature_names_out()

# 4. Calculating the mean TF-IDF score of each bigram across all non-depressive tweets
scores_double_ones = tfidf_matrix_double_ones.mean(axis=0).tolist()[0]

# 5. Creating a DataFrame with bigrams and their corresponding TF-IDF scores
importance_df_ones_double = pd.DataFrame({'word': feature_names_ones, 'score': scores_double_ones})

# 6. Sorting the DataFrame from most important to least important bigram
importance_df_ones_double = importance_df_ones_double.sort_values(by='score', ascending=False)

# Displaying the sorted DataFrame
print(importance_df_ones_double)

# Creating a bar chart for better visualization of bigram importance
plt.figure(figsize=(12,9))

# Horizontal bar plot showing bigram scores
sns.barplot(x='score', y='word', data=importance_df_ones_double)

# Labeling axes
plt.xlabel('Score_count')
plt.ylabel('Words')

# Adding a descriptive title
plt.title("Bar Chart of Importance for Non-Depression Classification (Bigrams)")

# Displaying the chart
plt.show()

# Now, let's check if these two bigram lists have any common sequences of words

# Creating an empty list to store common bigrams between depressive and non-depressive tweets
Common_biagram_words = []

# Iterating over each bigram in the non-depressive (positive) importance DataFrame
for words in importance_df_ones_double['word']:
    
    # Checking if the bigram also appears in the depressive (negative) importance DataFrame
    if words in importance_df_zero_double['word'].values:
        Common_biagram_words.append(words)

# Printing all common bigrams
print("Common biagram words are:")
print(Common_biagram_words)

# Printing the number of common bigrams
print(len(Common_biagram_words))
print('\n')

# Observation:
# Again, we notice that there are still many bigrams common between the two categories.

# To improve the analysis, we remove these common bigrams from consideration
# Creating lists of unique bigrams for each category
unique_pos_bigrams = [b for b in importance_df_ones_double['word'] if b not in Common_biagram_words]
unique_neg_bigrams = [b for b in importance_df_zero_double['word'] if b not in Common_biagram_words]

# Printing the unique bigrams for the positive (Not Depression) category
print("Unique positive words for Bigrams are:")
print(unique_pos_bigrams)
print('\n')

# Printing the unique bigrams for the negative (Depression) category
print("Unique negative words for Bigrams are:")
print(unique_neg_bigrams)

# ## Now,what we are going to do Actually is to work on some trigrams to see if they actually give any important help


# Firstly, let's explore trigrams (three-word sequences) to see if they provide meaningful insights

# Because some words are common between classes, analyzing n-grams beyond unigrams/bigrams
# can help capture context and clarify distinctions for each category

#Now, let's apply the process for Depressed people(0) using Trigrams.
# Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. Training the TF-IDF vectorizer with trigrams for depressive tweets
# stop_words='english' removes common words
# max_features=30 limits the vocabulary to the top 30 trigrams
# ngram_range=(3, 3) ensures only trigrams are extracted
vectorizer = TfidfVectorizer(stop_words='english', max_features=30, ngram_range=(3, 3))
tfidf_matrix_triple = vectorizer.fit_transform(zero_number)

# 3. Getting the feature names (trigrams)
feature_names_zero_triple = vectorizer.get_feature_names_out()

# 4. Calculating the mean TF-IDF score for each trigram across all depressive tweets
scores_zero_triple = tfidf_matrix_triple.mean(axis=0).tolist()[0]

# 5. Creating a DataFrame with trigrams and their corresponding TF-IDF scores
importance_df_zero_triple = pd.DataFrame({'word': feature_names_zero_triple, 'score': scores_zero_triple})

# 6. Sorting the DataFrame from most important to least important trigram
importance_df_zero_triple = importance_df_zero_triple.sort_values(by='score', ascending=False)

# Displaying the sorted DataFrame
print(importance_df_zero_triple)

# Creating a bar chart for better visualization of trigram importance
plt.figure(figsize=(12,9))

# Horizontal bar plot showing trigram scores
sns.barplot(x='score', y='word', data=importance_df_zero_triple)

# Labeling axes
plt.xlabel('Score_count')
plt.ylabel('Words')

# Adding a descriptive title
plt.title("Bar Chart of Importance for Depression Classification (Trigrams)")

# Displaying the chart
plt.show()

# Now, let's apply the same procedure for positive / Non-Depressed People (1) using trigrams

# Importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. Training the TF-IDF vectorizer with trigrams for non-depressive tweets
# stop_words='english' removes common words
# max_features=30 limits the vocabulary to the top 30 trigrams
# ngram_range=(3,3) ensures we extract only trigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=30, ngram_range=(3, 3))
tfidf_matrix_triple_ones = vectorizer.fit_transform(one_number)

# 3. Getting the feature names (trigrams)
feature_names_ones = vectorizer.get_feature_names_out()

# 4. Calculating the mean TF-IDF score of each trigram across all non-depressive tweets
scores_triple_ones = tfidf_matrix_triple_ones.mean(axis=0).tolist()[0]

# 5. Creating a DataFrame with trigrams and their corresponding TF-IDF scores
importance_df_ones_triple = pd.DataFrame({'word': feature_names_ones, 'score': scores_triple_ones})

# 6. Sorting the DataFrame from most important to least important trigram
importance_df_ones_triple = importance_df_ones_triple.sort_values(by='score', ascending=False)

# Displaying the sorted DataFrame
print(importance_df_ones_triple)

# Creating a bar chart for better visualization of trigram importance
plt.figure(figsize=(12,9))

# Horizontal bar plot showing trigram scores
sns.barplot(x='score', y='word', data=importance_df_ones_triple)

# Labeling axes
plt.xlabel('Score_count')
plt.ylabel('Words')

# Adding a descriptive title
plt.title("Bar Chart of Importance for Non-Depression Classification (Trigrams)")

# Displaying the chart
plt.show()

# We explored trigrams to capture richer context, but they significantly increased dimensionality and did not provide meaningful performance improvement compared to unigrams and bigrams. Therefore, we decided to proceed with (1,2) n-grams for the final model.


# 


# ## 5️⃣ Model Training - Evaluation


# ## 1.Logistic Regression


# Building a Logistic Regression model to classify depressive vs non-depressive tweets

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics
from nltk.corpus import stopwords

# 1. Preparing the stopwords list
# Combine standard English stopwords with custom neutral words identified earlier
final_stopwords = list(set(stopwords.words('english')).union(set(custom_stopwords)))

# 2. Splitting the dataset into training and testing sets
# stratify=y ensures the class distribution remains the same in both sets
X = Final['text_lemmatized']
y = Final['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Converting text into numerical features using TF-IDF
# Using both unigrams and bigrams to capture single words and word pairs
# Limiting to 10,000 features for efficiency
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words=final_stopwords)
X_train_tfidf_logistic = vectorizer.fit_transform(X_train)
X_test_tfidf_logistic = vectorizer.transform(X_test)

# 4. Training the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Training the model on the TF-IDF features
model.fit(X_train_tfidf_logistic, y_train)

# Displaying the class labels
print(model.classes_)

# 5. Evaluating the model
# Predicting on the test set
y_pred = model.predict(X_test_tfidf_logistic)

# Printing classification metrics: precision, recall, f1-score
print(classification_report(y_test, y_pred))

# Creating a confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Displaying the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0,1])
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(
    model, 
    X_test_tfidf_logistic, 
    y_test, 
    cmap='pink', 
    display_labels=['Depression', 'No Depression']
)
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# The logistic regression model achieved an overall accuracy of 0.89 on the test set, indicating strong predictive performance. Both classes performed similarly, with class 0 having a precision of 0.86 and recall of 0.92, and class 1 having a precision of 0.92 and recall of 0.86. The balanced F1-scores of 0.89 for both classes suggest the model maintains a good trade-off between precision and recall across the dataset.


# ## 2.Random Forest


# Applying Random Forest Classifier to classify depressive vs non-depressive tweets

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Splitting dataset into Train and Test sets
# stratify=y ensures class distribution remains consistent
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Converting text into numerical features using TF-IDF (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words=final_stopwords)
X_train_tfidf_random = vectorizer.fit_transform(X_train_random)
X_test_tfidf_random = vectorizer.transform(X_test_random)

# 3. Training the Random Forest Classifier
# n_jobs=-1 uses all CPU cores for faster training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=25, n_jobs=-1)
rf_classifier.fit(X_train_tfidf_random, y_train_random)

# 4. Predicting on the Test set
y_pred_random = rf_classifier.predict(X_test_tfidf_random)

# 5. Evaluation
accuracy = accuracy_score(y_test_random, y_pred_random)
classification_rep = classification_report(y_test_random, y_pred_random)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# --- Sample Prediction Demonstration ---
# Selecting the first tweet from the test set
sample_random = X_test_random.iloc[0:1]

# Transforming the sample text using the trained TF-IDF vectorizer
sample_tfidf_random = vectorizer.transform(sample_random) 

# Making prediction using the Random Forest model
prediction = rf_classifier.predict(sample_tfidf_random)

print(f"\nSample Text: {sample_random.values[0]}")
print(f"Prediction: {prediction[0]}")

# 6. Displaying the Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(
    rf_classifier, 
    X_test_tfidf_random, 
    y_test_random, 
    cmap='Oranges', 
    display_labels=['Depression', 'No Depression']
)
plt.title("Confusion Matrix: Random Forest")
plt.show()

# The Random Forest model achieved an overall accuracy of 0.85 on the test set, showing solid predictive performance. Class 0 had a precision of 0.83 and recall of 0.88, while class 1 had a precision of 0.88 and recall of 0.83, indicating fairly balanced performance across classes. The F1-scores of 0.86 for class 0 and 0.85 for class 1 suggest the model maintains a good trade-off between precision and recall,


# ## 3.Support Vector Machines(SVM)


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Διαχωρισμός σε Train και Test σετ (Αν δεν το έχεις κάνει ήδη σε αυτό το cell)
X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Μετατροπή κειμένου σε TF-IDF (Αν δεν είναι ήδη έτοιμα τα X_train_tfidf)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words=final_stopwords)
X_train_tfidf_SVM = vectorizer.fit_transform(X_train_SVM)
X_test_tfidf_SVM = vectorizer.transform(X_test_SVM)

# 3. Εκπαίδευση του LinearSVC (Πολύ πιο γρήγορο από το SVC)
# Το C=1.0 είναι η standard ρύθμιση.
clf_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
clf_svc.fit(X_train_tfidf_SVM, y_train_SVM)

# 4. Πρόβλεψη και Αξιολόγηση
y_pred_svc = clf_svc.predict(X_test_tfidf_SVM)

print(f"Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")
print("\nClassification Report (SVM):")
print(classification_report(y_test_SVM, y_pred_svc))

# 5. Confusion Matrix για να δεις τα λάθη
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(clf_svc, X_test_tfidf_SVM, y_test_SVM, cmap='Purples', display_labels=['Depression', 'No'])
plt.title("Confusion Matrix: Linear SVM")
plt.show()

# The SVM model achieved an overall accuracy of 0.88 on the test set, demonstrating strong predictive performance. Class 0 had a precision of 0.86 and recall of 0.90, while class 1 had a precision of 0.90 and recall of 0.86, showing balanced performance across both classes. With F1-scores of 0.88 for both classes, the model maintains an effective trade-off between precision and recall, making it reliable for classification tasks.


# ## 4.XGBoost Algorithm


import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# 1. Διαχωρισμός σε Train και Test σετ (Αν δεν το έχεις κάνει ήδη σε αυτό το cell)
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Μετατροπή κειμένου σε TF-IDF (Αν δεν είναι ήδη έτοιμα τα X_train_tfidf)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words=final_stopwords)
X_train_tfidf_boost = vectorizer.fit_transform(X_train_boost)
X_test_tfidf_boost = vectorizer.transform(X_test_boost)
# 3. Δημιουργία και εκπαίδευση του μοντέλου XGBoost
# Χρησιμοποιούμε τον XGBClassifier που είναι συμβατός με το scikit-learn
model_boost = xgb.XGBClassifier(
    n_estimators=100,       # Αριθμός δέντρων
    max_depth=3,            # Μέγιστο βάθος δέντρου
    learning_rate=0.1,      # Ρυθμός μάθησης
    objective='binary:logistic', # Στόχος: δυαδική ταξινόμηση
    random_state=42
)

model_boost.fit(X_train_tfidf_boost, y_train_boost)

# 4. Προβλέψεις στα δεδομένα δοκιμής
y_pred_boost = model_boost.predict(X_test_tfidf_boost)

# 5. Αξιολόγηση του μοντέλου
accuracy = accuracy_score(y_test_boost, y_pred_boost)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_boost, y_pred_boost))
# 5. Confusion Matrix για να δεις τα λάθη
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(model_boost, X_test_tfidf_boost, y_test_boost, cmap='Greens', display_labels=['Depression', 'No'])
plt.title("Confusion Matrix: XGBoost")
plt.show()

# The XGBoost model achieved an overall accuracy of 78.6% on the test set, reflecting moderate predictive performance. Class 0 had high recall (0.93) but lower precision (0.72), while class 1 showed high precision (0.90) but lower recall (0.65), indicating the model is better at identifying class 0 than class 1. The F1-scores of 0.81 for class 0 and 0.76 for class 1 highlight a reasonable balance between precision and recall, though performance on class 1 is somewhat weaker.


# ## 5.NAIVE BAYES Algorithm


# Applying Naive Bayes classification **only for Unigrams + Bigrams** (1,2 n-grams)

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

results = {}

# Only using Bigrams (unigrams + bigrams)
name = "Unigrams+Bigrams"
params = {"ngram_range": (1, 2)}

print(f"\n{'='*50}")
print(f"  Naive Bayes with {name}")
print(f"{'='*50}")

# TF-IDF vectorization using only unigrams+bigrams
vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    ngram_range=params["ngram_range"],
    max_features=10000,
    sublinear_tf=True
)

# Fit-transform the training data and transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes model
model_naive = MultinomialNB()
model_naive.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model_naive.predict(X_test_tfidf)

# Compute accuracy and store results
acc = accuracy_score(y_test, y_pred)
results[name] = {"accuracy": acc, "y_pred": y_pred, "model": model_naive, "vectorizer": vectorizer}

# Print metrics
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Depressive (0)", "Not Depressive (1)"]
))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Depressive", "Not Depressive"],
            yticklabels=["Depressive", "Not Depressive"])
plt.title(f"Confusion Matrix - Naive Bayes ({name})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ============================================================
# STEP: TOP DISCRIMINATIVE WORDS FOR EACH CLASS
# ============================================================

best_name       = name
best_model      = results[best_name]["model"]
best_vectorizer = results[best_name]["vectorizer"]

feature_names = best_vectorizer.get_feature_names_out()
log_probs     = best_model.feature_log_prob_

top_n = 15
top_dep_idx     = np.argsort(log_probs[0])[-top_n:][::-1]
top_not_dep_idx = np.argsort(log_probs[1])[-top_n:][::-1]

top_dep_words     = [(feature_names[i], log_probs[0][i]) for i in top_dep_idx]
top_not_dep_words = [(feature_names[i], log_probs[1][i]) for i in top_not_dep_idx]

# Plotting top discriminative words for each class
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

words_d, scores_d = zip(*top_dep_words)
axes[0].barh(words_d[::-1], scores_d[::-1], color="red", alpha=0.7)
axes[0].set_title(f"Top {top_n} Words → Depressive (0)\n(Naive Bayes {best_name})")
axes[0].set_xlabel("Log Probability")

words_n, scores_n = zip(*top_not_dep_words)
axes[1].barh(words_n[::-1], scores_n[::-1], color="green", alpha=0.7)
axes[1].set_title(f"Top {top_n} Words → Not Depressive (1)\n(Naive Bayes {best_name})")
axes[1].set_xlabel("Log Probability")

plt.tight_layout()
plt.show()

print(f"\n✔ Analysis completed! Configuration used: {best_name}")

# The Naive Bayes model with unigrams and bigrams achieved an overall accuracy of 78.2% on the test set, showing moderate predictive performance. The Depressive class (0) had a precision of 0.81 and recall of 0.73, while the Not Depressive class (1) had a precision of 0.76 and recall of 0.83, indicating complementary strengths in detecting each class. The F1-scores of 0.77 for class 0 and 0.80 for class 1 suggest the model maintains a reasonable balance between precision and recall across the dataset.
# 
# The log probability analysis shows that the Naive Bayes model can pick out key words for each class, with negative words like "hate" and "sadness" appearing in the depressive category, and positive words like "great" and "happy" in the non-depressive category. These keywords strongly relate to the target sentiment. However, the overall accuracy of 78.2% shows that, while helpful, unigrams and bigrams can’t capture all the overlap in language between the classes.


# ## 6️⃣ Conclusions


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Logistic Regression probabilities
y_pred_proba_logistic = model.predict_proba(X_test_tfidf_logistic)[:, 1]

# 2️⃣ Random Forest probabilities
y_pred_proba_random = rf_classifier.predict_proba(X_test_tfidf_random)[:, 1]

# 3️⃣ XGBoost probabilities
y_pred_proba_boost = model_boost.predict_proba(X_test_tfidf_boost)[:, 1]

# 4️⃣ Linear SVM decision function scores
y_score_svc = clf_svc.decision_function(X_test_tfidf_SVM)

# 5️⃣ Naive Bayes probabilities (Bigrams)
X_test_tfidf_nb = results[best_name]["vectorizer"].transform(X_test)  # Use the NB vectorizer
y_pred_proba_nb = results[best_name]["model"].predict_proba(X_test_tfidf_nb)[:, 1]

# -------------------------------
# Compute ROC curves and AUC
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_proba_logistic)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

fpr_random, tpr_random, _ = roc_curve(y_test_random, y_pred_proba_random)
roc_auc_random = auc(fpr_random, tpr_random)

fpr_boost, tpr_boost, _ = roc_curve(y_test, y_pred_proba_boost)
roc_auc_boost = auc(fpr_boost, tpr_boost)

fpr_svc, tpr_svc, _ = roc_curve(y_test_SVM, y_score_svc)
roc_auc_svc = auc(fpr_svc, tpr_svc)

fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_proba_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# -------------------------------
# Plot ROC curves for all 5 models
plt.figure(figsize=(10, 8))
plt.plot(fpr_logistic, tpr_logistic, color='darkblue', lw=2,
         label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.plot(fpr_random, tpr_random, color='green', lw=2,
         label=f'Random Forest (AUC = {roc_auc_random:.2f})')
plt.plot(fpr_boost, tpr_boost, color='orange', lw=2,
         label=f'XGBoost (AUC = {roc_auc_boost:.2f})')
plt.plot(fpr_svc, tpr_svc, color='purple', lw=2,
         label=f'Linear SVM (AUC = {roc_auc_svc:.2f})')
plt.plot(fpr_nb, tpr_nb, color='black', lw=2,
         label=f'Naive Bayes (Bigrams, AUC = {roc_auc_nb:.2f})')

# Random baseline
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='No Skill (Random)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC - AUC Curve for All 5 Models')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Δημιουργία λίστας για αποθήκευση αποτελεσμάτων
results_table = []

# -------------------------------
# 1️⃣ Logistic Regression
results_table.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, model.predict(X_test_tfidf_logistic)),
    "Precision": precision_score(y_test, model.predict(X_test_tfidf_logistic)),
    "Recall": recall_score(y_test, model.predict(X_test_tfidf_logistic)),
    "F1-Score": f1_score(y_test, model.predict(X_test_tfidf_logistic)),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba_logistic)
})

# -------------------------------
# 2️⃣ Random Forest
results_table.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test_random, rf_classifier.predict(X_test_tfidf_random)),
    "Precision": precision_score(y_test_random, rf_classifier.predict(X_test_tfidf_random)),
    "Recall": recall_score(y_test_random, rf_classifier.predict(X_test_tfidf_random)),
    "F1-Score": f1_score(y_test_random, rf_classifier.predict(X_test_tfidf_random)),
    "ROC-AUC": roc_auc_score(y_test_random, y_pred_proba_random)
})

# -------------------------------
# 3️⃣ Linear SVM
results_table.append({
    "Model": "Linear SVM",
    "Accuracy": accuracy_score(y_test_SVM, clf_svc.predict(X_test_tfidf_SVM)),
    "Precision": precision_score(y_test_SVM, clf_svc.predict(X_test_tfidf_SVM)),
    "Recall": recall_score(y_test_SVM, clf_svc.predict(X_test_tfidf_SVM)),
    "F1-Score": f1_score(y_test_SVM, clf_svc.predict(X_test_tfidf_SVM)),
    "ROC-AUC": roc_auc_score(y_test_SVM, y_score_svc)
})

# -------------------------------
# 4️⃣ Naive Bayes (Unigrams or Uni+Bigrams depending on best_name)
X_test_tfidf_nb = results[best_name]["vectorizer"].transform(X_test)
nb_model = results[best_name]["model"]

results_table.append({
    "Model": f"Naive Bayes ({best_name})",
    "Accuracy": accuracy_score(y_test, nb_model.predict(X_test_tfidf_nb)),
    "Precision": precision_score(y_test, nb_model.predict(X_test_tfidf_nb)),
    "Recall": recall_score(y_test, nb_model.predict(X_test_tfidf_nb)),
    "F1-Score": f1_score(y_test, nb_model.predict(X_test_tfidf_nb)),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba_nb)
})

# -------------------------------
# 5️⃣ XGBoost
results_table.append({
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test, model_boost.predict(X_test_tfidf_boost)),
    "Precision": precision_score(y_test, model_boost.predict(X_test_tfidf_boost)),
    "Recall": recall_score(y_test, model_boost.predict(X_test_tfidf_boost)),
    "F1-Score": f1_score(y_test, model_boost.predict(X_test_tfidf_boost)),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba_boost)
})

# -------------------------------
# Δημιουργία DataFrame
results_df = pd.DataFrame(results_table)

# Ταξινόμηση κατά ROC-AUC
results_df = results_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)


results_df = results_df.round(3)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))

plt.title("Model Comparison Results", pad=20)
plt.tight_layout()
plt.show()

# Logistic Regression is the superior model in this evaluation, as it achieves the highest Accuracy (88.9%), Precision (91.9%), and ROC-AUC (0.95), consistently outperforming the other four models across nearly every metric. While the Linear SVM follows closely as a strong second, Logistic Regression provides the best overall balance for distinguishing between the classes.
