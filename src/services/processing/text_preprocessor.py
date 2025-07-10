import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import contractions

class TextPreprocessor:
    __lemmatizer__ = WordNetLemmatizer()
    __stop_words__ = set(stopwords.words('english'))
    __instance__ = None

    @staticmethod
    def getInstance():
        if TextPreprocessor.__instance__ == None:
            TextPreprocessor.__instance__ = TextPreprocessor()
        return TextPreprocessor.__instance__

    def __clean_text__(self, text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Convert to lowercase
        text = contractions.fix(text)

        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text

    def __remove_stopwords__(self, text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.__stop_words__]
        return ' '.join(filtered_words)

    def __get_wordnet_pos__(self, tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    def __lemmatize_text__(self, text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [self.__lemmatizer__.lemmatize(word, pos=self.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    def preprocess_text(self, text, remove_stopwords_flag=True): 
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return [] 
        
       
        text = self.__clean_text__(text)
        text = self.__lemmatize_text__(text)
        
      
        if remove_stopwords_flag:
            text = self.__remove_stopwords__(text)

        return word_tokenize(text.strip())
