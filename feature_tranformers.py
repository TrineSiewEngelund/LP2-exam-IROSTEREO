from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import demoji
import numpy as np
import unicodedata
import re
import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
sentiment_analyzer = SentimentIntensityAnalyzer()




### Preprocessing of scalar features  ###
class preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        docs = []
        for user in X:
            doc = ' '.join(user) # join all tweets to one doc
            doc = doc[:]
            doc = unicodedata.normalize('NFKC', doc) # normalize to unicode
            doc = re.sub(r"#(USER|URL|HASHTAG)+#", '', str(doc)) # remove tags
            doc = re.sub(r'[^a-zA-Z0-9\s\'\‘\’]', ' ', str(doc)) # substitute everything that's not letters, digits, spaces or one of the following: '‘’ with a space
            doc = re.sub(r'\s\s+', ' ', str(doc)) # substitute multiple spaces with one space
            docs.append(doc)
        return np.array(docs)




### Preprocessing of vector-based features  ###
class empty2dot(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = []
        for user in X:
            user = user[:]
            new_user=[words if words!='' else '.' for words in user] 
            new_X.append(new_user)
        return np.array(new_X)




### Stylometric features ###

# list of emojis
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

class stylometric_counts(BaseEstimator, TransformerMixin):
    '''
        Input: matrix with shape (-1,)
        Output: matrix with shape (-1,10)
        Counts "USER", "URL", "HASHTAG", 5 specific emojis, all emojis, multiple kinds of laughs and repetition of punctuation. 

    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array([' '.join(user_tweets) for user_tweets in X]) # Joins a user's tweets into one doc

        # Initiate empty lists
        user_count, url_count, hashtag_count, emoji_slightly_smile_count, emoji_laugh_tears_count, emoji_lol_count, emoji_upside_count, thumbs_up_count, emojis_count_all, laugh_count, repeat_punctuation_count = [], [], [], [], [], [], [], [], [], [], []

        for user_tweets in X:
            user_tweets = user_tweets[:].lower()

            # Count tags
            user = user_tweets.count("#user#") 
            user_count.append(user)   
    
            url = user_tweets.count("#url#")
            url_count.append(url)
    
            hashtag = user_tweets.count("#hashtag#")
            hashtag_count.append(hashtag)

            # Count specific emojis
            emoji_slightly_smile = user_tweets.count('🙂')
            emoji_slightly_smile_count.append(emoji_slightly_smile)

            emoji_laugh_tears = user_tweets.count('😂')
            emoji_laugh_tears_count.append(emoji_laugh_tears)

            emoji_lol = user_tweets.count('🤣')
            emoji_lol_count.append(emoji_lol)

            emoji_upside = user_tweets.count('🙃')
            emoji_upside_count.append(emoji_upside)

            emoji_thumbs_up = user_tweets.count('👍')
            thumbs_up_count.append(emoji_thumbs_up)

            # Count all emojis
            all_emojis = ''.join(EMOJI_PATTERN.findall(user_tweets))
            emojis_count_all.append(len(all_emojis))

            # Count laughing expressions
            lols = re.findall(r'\b(?:l(?:o|e|aw))+l*z*\b', str(user_tweets)) # Count various versions of lol
            lmaos = re.findall(r'\blmf*ao*\b', str(user_tweets)) # Count various versions of lm(f)ao
            rotfls = re.findall(r'\brotfl(?:ol)*\b', str(user_tweets)) # Count various versions of rotfl
            hahas = re.findall(r'\b(?:b|g|mu|mw)*(?:a|e|ee)*h*(?:h(?:a|e|ee))+h*(?:a|e|ee)*\b', str(user_tweets)) # Count various versions of haha (e.g. bahaha, ahha etc.)
            emoticons = re.findall(r'(?::|X|x|=)\'*-*(?:\)|D|d)*', str(user_tweets)) # Count various versions of emoticons (e.g. XD :'D :DD etc.)
            laughs = len(lols+lmaos+rotfls+hahas+emoticons)
            laugh_count.append(laughs)

            # Count use of repeated punctuation (e.g. !?? or !!!)
            repeat_punctuation = len(re.findall(r'(?:\!\!+|\?\?+|\.\.+|\?\!|\!\?)', str(user_tweets)))
            repeat_punctuation_count.append(repeat_punctuation)

        # Create and reshape arrays
        user_count = np.array(user_count).reshape((-1,1)) 
        url_count = np.array(url_count).reshape((-1,1))
        hashtag_count = np.array(hashtag_count).reshape((-1,1))
        emoji_slightly_smile_count = np.array(emoji_slightly_smile_count).reshape((-1,1))
        emoji_laugh_tears_count = np.array(emoji_laugh_tears_count).reshape((-1,1))
        emoji_lol_count = np.array(emoji_lol_count).reshape((-1,1))
        emoji_upside_count = np.array(emoji_upside_count).reshape((-1,1))
        thumbs_up_count = np.array(thumbs_up_count).reshape((-1,1))
        emojis_count_all = np.array(emojis_count_all).reshape((-1,1))
        laugh_count = np.array(laugh_count).reshape((-1,1))
        repeat_punctuation_count = np.array(repeat_punctuation_count).reshape((-1,1))
  
        return np.hstack((user_count,url_count, hashtag_count, emoji_slightly_smile_count, emoji_laugh_tears_count, emoji_lol_count, emoji_upside_count, thumbs_up_count, emojis_count_all, laugh_count, repeat_punctuation_count))




### SpongeBob ###
class spongebob(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_spongebob_ratio = []
        for user in X:
            user_demoji = demoji.replace(user, "") # substitute emojis with nothing, i.e. remove them from tweet
            mixed_words = sum(1 for word in user_demoji.split() if not word.isupper() and not word.islower() and not word.istitle())

            if len(user.split()) > 0: # to avoid division by zero-error
                proportion_mixed = mixed_words/len(user.split())

            X_spongebob_ratio.append(proportion_mixed)
        
        return np.array(X_spongebob_ratio).reshape((-1, 1))




### Lexical diversity (TTR) ###

# helper function to transformer
def ttr_func(X):
    stemmer = PorterStemmer()
    tokens = word_tokenize(X)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    V = len(set(stemmed_tokens))
    N = len(stemmed_tokens)
    return V/N

# transformer
class TTR(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_ttr_ratio = []
        for user in X:
            user = user[:]
            X_ttr_ratio.append(ttr_func(user))
        
        return np.array(X_ttr_ratio).reshape((-1,1))
        #return np.array([ratio(tweets[:]) for tweets in X]).reshape((-1,1))




### Average no. of characters per tweet ###
class avg_char(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X_avg_char = []
        for user in X:
            user = user[:]
            X_avg_char.append(len(user)/200)
        
        return np.array(X_avg_char).reshape((-1,1))
        #return np.array([len(tweets[:])/200 for tweets in X]).reshape((-1,1))




### Average no. of words per tweet ###
class avg_word(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_avg_word = []
        for user in X:
            user = user[:]
            X_avg_word.append(len(word_tokenize(user))/200)

        return np.array(X_avg_word).reshape(-1, 1)
        # return np.array([len(word_tokenize(tweets[:]))/200 for tweets in X]).reshape((-1,1))




### Emoji sentiment difference ###
class emoji_sentiment_diff(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_senti_diff = []

        for user in X:
            user_senti_diff = []

            for tweets in user:
                tweets = tweets[:]
                # emojis (without text) and its sentiment / polarity score
                emojis_without = ''.join(EMOJI_PATTERN.findall(str(tweets)))
                emoji_sentiment = np.array(sentiment_analyzer.polarity_scores(emojis_without)["compound"])

                # text (without emojis) and its sentiment / polarity score
                tweets_without = EMOJI_PATTERN.sub(r' ', str(tweets)) # Substitute all emojis with a space
                tweets_without = re.sub(r'\s\s+', ' ', str(tweets_without)) # substitute multiple spaces with one space
                tweets_sentiment = np.array(sentiment_analyzer.polarity_scores(tweets_without)["compound"])

                # difference in polarity scores
                tweets_senti_diff = abs(tweets_sentiment - emoji_sentiment)

                user_senti_diff.append(tweets_senti_diff)

            X_senti_diff.append(user_senti_diff)
        
        return np.array(X_senti_diff)





### Sentiment incongruity ###

# helper function
def pos_vs_neg(sent, sent_analyzer):
    """
    Finds the absolute difference between 
    the most positive and negative expression in a sentence.
    """
    tokens = word_tokenize(sent)
    scores = [sent_analyzer.polarity_scores(word)['compound'] for word in tokens]
    return abs(max(scores)-min(scores))

class sentiment_incongruity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Returns the sentiment incongruity score 
        (absolute difference between the most positive and negative expression)
        for each tweet.
        """
    
        X_senti_diff = []

        for user in X:
            user = user[:]
            word_senti_diff = [pos_vs_neg(tweet, sentiment_analyzer) for tweet in user]
            X_senti_diff.append(word_senti_diff)
    
        return np.array(X_senti_diff)