import re
import contractions
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class AntiqueTextProcessor:
    __lemmatizer__ = WordNetLemmatizer()

    @staticmethod
    def __clean_text__(text):
        """
        Clean text by removing special characters and converting to lowercase
        """

        # Expand contractions
        text = contractions.fix(text)

        # Convert to lowercase
        text = text.lower()
        
        return text

    @staticmethod
    def __get_wordnet_pos__(tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def __lemmatize_text__(text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [AntiqueTextProcessor.__lemmatizer__.lemmatize(word, pos=AntiqueTextProcessor.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    @staticmethod
    def preprocess_text(text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Apply all preprocessing steps
        text = AntiqueTextProcessor.__clean_text__(text)
        text = AntiqueTextProcessor.__lemmatize_text__(text)
        text = word_tokenize(text.strip())

        return text

class QuoraTextProcessor:
    __lemmatizer__ = WordNetLemmatizer()
    __stop_words__ = set(stopwords.words('english'))

    @staticmethod
    def __clean_text__(text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s\d]', '', text)
        
        return text

    @staticmethod
    def __remove_stopwords__(text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in QuoraTextProcessor.__stop_words__]
        return ' '.join(filtered_words)

    @staticmethod
    def __get_wordnet_pos__(tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def __lemmatize_text__(text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [QuoraTextProcessor.__lemmatizer__.lemmatize(word, pos=QuoraTextProcessor.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    @staticmethod
    def preprocess_text(text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Apply all preprocessing steps
        text = QuoraTextProcessor.__clean_text__(text)
        text = QuoraTextProcessor.__remove_stopwords__(text)
        text = QuoraTextProcessor.__lemmatize_text__(text)

        return text.strip().split()

class WebisTextProcessor:
    __lemmatizer__ = WordNetLemmatizer()
    __stop_words__ = set(stopwords.words('english'))

    @staticmethod
    def __clean_text__(text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s\d]', '', text)
        
        return text

    @staticmethod
    def __remove_stopwords__(text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in WebisTextProcessor.__stop_words__]
        return ' '.join(filtered_words)

    @staticmethod
    def __get_wordnet_pos__(tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def __lemmatize_text__(text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [WebisTextProcessor.__lemmatizer__.lemmatize(word, pos=WebisTextProcessor.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    @staticmethod
    def preprocess_text(text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Apply all preprocessing steps
        text = WebisTextProcessor.__clean_text__(text)
        text = WebisTextProcessor.__remove_stopwords__(text)
        text = WebisTextProcessor.__lemmatize_text__(text)

        return text.strip().split()

class RecreationTextProcessor:
    __lemmatizer__ = WordNetLemmatizer()
    __stop_words__ = set(stopwords.words('english'))

    @staticmethod
    def __clean_text__(text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s\d]', '', text)
        
        return text

    @staticmethod
    def __remove_stopwords__(text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in RecreationTextProcessor.__stop_words__]
        return ' '.join(filtered_words)

    @staticmethod
    def __get_wordnet_pos__(tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def __lemmatize_text__(text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [RecreationTextProcessor.__lemmatizer__.lemmatize(word, pos=RecreationTextProcessor.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    @staticmethod
    def preprocess_text(text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Apply all preprocessing steps
        text = RecreationTextProcessor.__clean_text__(text)
        text = RecreationTextProcessor.__remove_stopwords__(text)
        text = RecreationTextProcessor.__lemmatize_text__(text)

        return text.strip().split()

class WikirTextProcessor:
    __lemmatizer__ = WordNetLemmatizer()
    __stop_words__ = set(stopwords.words('english'))

    @staticmethod
    def __clean_text__(text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Expand contractions
        text = contractions.fix(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s\d]', '', text)
        
        return text

    @staticmethod
    def __remove_stopwords__(text):
        """
        Remove common stopwords from text
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in WikirTextProcessor.__stop_words__]
        return ' '.join(filtered_words)

    @staticmethod
    def __get_wordnet_pos__(tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
            }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def __lemmatize_text__(text):
        """
        Lemmatize words to their root form
        """
        # Tokenize into words
        words = word_tokenize(text)

        # POS tagging
        pos_tags = pos_tag(words)

        lemmatized_words = [WikirTextProcessor.__lemmatizer__.lemmatize(word, pos=WikirTextProcessor.__get_wordnet_pos__(tag)) for word, tag in pos_tags]

        return ' '.join(lemmatized_words)

    @staticmethod
    def preprocess_text(text):
        """
        Apply full preprocessing pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Apply all preprocessing steps
        text = WikirTextProcessor.__clean_text__(text)
        text = WikirTextProcessor.__remove_stopwords__(text)
        text = WikirTextProcessor.__lemmatize_text__(text)

        return text.strip().split()
