from typing import Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
# If you don't download it, you'll get an error like:
# LookupError: Resource punkt not found.
# Please use the NLTK Downloader to obtain the resource.
nltk.download('punkt_tab')


def clean_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    return ' '.join(text.split())

def extract_metadata(text: str) -> Dict: # "Hello there! How are you? I'm good."
    sentences = sent_tokenize(text) # Split text into sentences: ["Hello there!", "How are you?", "I'm good."]
    words = word_tokenize(text) # Split text into words: ["Hello", "there", "!", "How", "are", "you", "?", "I", "'m", "good", "."]
    
    return {
        'num_sentences': len(sentences),
        'num_words': len(words)
    }

def preprocess(text: str) -> Dict:
    processed_text = text
    metadata = {}

    processed_text = clean_text(processed_text)

    metadata = extract_metadata(processed_text)

    return {
        'processed_text': processed_text,
        'metadata': metadata
    }