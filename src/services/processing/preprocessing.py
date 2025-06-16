import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text

    def remove_stopwords(self, text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def get_wordnet_pos(self, tag_parameter):

        tag = tag_parameter[0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_text(self, text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [self.lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in pos_tags]
        
        return ' '.join(lemmatized_words)

    def preprocess_text(self, text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
            
        # Apply all preprocessing steps
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)

        return text.strip()
