import regex
from icu import Locale, BreakIterator

class ICUWordPreTokenizer:
    """ A Word PreTokenizer with pyICU """
    def __init__(self, stopword_sets: set[str] = {}):
        self.word_iterator = BreakIterator.createWordInstance(Locale("en"))
        self.re_bad_chars = regex.compile(r"[\p{Cc}\p{Cs}\p{Cn}]+")
        self.stopword_sets = stopword_sets
    
    def __call__(self, text: str, remove_stopwords: bool=False, lowercase: bool=True) -> list[str]:
        """ Word tokenize. Split the word boundary with ICU
            Args:
                text (str): String.
            Returns:
                (list[str]) List of word splitted from text. Stop words and punctions will be removed.
            Note:
                Returns may be empty list.
        """
        assert isinstance(text, str)
        # Fix non-utf8 error: https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790
        # Note: Add `'Cn' -> Noncharacter` according to latest unicode category
        text = self.re_bad_chars.sub("", text).strip()

        # Fix ValueError: This Sequence is Empty
        if not text:
            return []
        
        if lowercase:
            text = text.lower()

        # Use ICU word boundary iterator
        self.word_iterator.setText(text)

        words = []
        start = 0
        for idx in self.word_iterator:
            word = text[start: idx].strip()
            if word:
                if remove_stopwords: 
                    if word not in self.stopword_sets:
                        words.append(word)
                else:
                    words.append(word)
                
            start = idx
        
        return words
