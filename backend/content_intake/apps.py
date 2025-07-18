import spacy
from django.apps import AppConfig
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

keyword_model = None
sentence_model = None
nlp_model_spaCy = None
class ContentIntakeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'content_intake'

    def ready(self):
        """
        Perform model loads once
        """
        global nlp_model_spaCy 
        global sentence_model
        global keyword_model

        if (nlp_model_spaCy == None):
            try:
                # Attempt to load the large English language model 
                nlp_model_spaCy = spacy.load('en_core_web_lg')
                print('spaCy model "en_core_web_lg" loaded successfully!')

                # Load sentence model 
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                keyword_model = KeyBERT(model=sentence_model)
                print('Keyword model "all-MiniLM-L6-v2" loaded successfully!')

            except OSError:
                print('Error: Language model "en_core_web_lg" not found.')
                print('Please ensure you have run "python -m spacy download en_core_web_lg" in your active Conda environment.')
            except Exception as e:
                print(f'Unexpected error occurred: { e }')
