import spacy
from django.apps import AppConfig

nlp_model_spaCy = None
class ContentIntakeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'content_intake'

    def ready(self):
        """
        Perform spaCy model load once
        """
        global nlp_model_spaCy 

        if (nlp_model_spaCy == None):
            try:
                # Attempt to load the large English language model 
                nlp_model_spaCy = spacy.load('en_core_web_lg')
                print('spaCy model "en_core_web_lg" loaded successfully!')
            except OSError:
                print('Error: Language model "en_core_web_lg" not found.')
                print('Please ensure you have run "python -m spacy download en_core_web_lg" in your active Conda environment.')
            except Exception as e:
                print(f'Unexpected error occurred: { e }')
