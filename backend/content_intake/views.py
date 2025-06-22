import numpy as np
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from .apps import nlp_model_spaCy

# Create your views here.
def process_content(request):

    # Step 1: 
    # Text extraction and cleaning - to be replaced with Rishie's code, using dummy text for now
    # Article link for text below -> https://www.nytimes.com/2025/06/20/world/canada/canada-banff-national-park-rock-slide.html 
    request = """
        Two people were killed and three others were injured after a cascade of sliding rocks fell on them on Thursday while they were hiking in Banff National Park in Alberta, one of Canada’s major tourism destinations.
        The authorities recovered the body of one of the two hikers who were killed on Thursday, a 70-year-old woman from Calgary. The authorities did not release her name.
        The second body was recovered on Friday[ the authorities have not provided more information about that victim. In total, 13 people, including bystanders, were removed by emergency crews.
        Officials said they were not looking for any other possible victims.
        The rockslide happened near Bow Glacier Falls, an area about a half-hour drive north of Lake Louise, a major attraction in the park known for its vibrant blue waters set against a majestic background of towering mountains. 
        Banff is Canada’s most popular national park, with about four million annual visitors.
        “Sadly, this extremely rare event appears to have been neither preventable nor predictable,” François Masse, a superintendent with Parks Canada, told reporters on Friday afternoon.
        Mr. Masse said it is common for water to enter fissures in mountains and to expand those cracks as it freezes into ice, causing chunks of rock to tumble down, including near this portion of the trail.
        But the unusual part of this case was the size of the rock that fell, Mr. Masse said. 
        Officials are still working to figure out how big the rock slab was, noting there were rainy conditions in the days before the incident. 
        Thursday’s weather was sunny, but the area is now expecting snowfall, officials said.
        Emergency crews were called to the area early Thursday afternoon, and found one hiker dead at the scene, which is part of a hiking trail. Two others were taken to a hospital by helicopters and the third in an ambulance. 
        The authorities did not disclose their injuries.
        The second hiker’s body was also found on Thursday but could not be safely recovered until Friday.
        Disaster response teams had been flying aircraft in the park on Friday using infrared cameras, which capture thermal imaging, to survey the grounds, but determined that there were no other victims.
        The authorities were working to notify the next of kin of the people killed in the rockslide and have dispatched a geotechnical engineer to survey the slope to determine the risks of more rockslides, according to a joint statement by Parks Canada and the Royal Canadian Mounted Police.
        The Bow Glacier Falls trail, where the hikers were traveling, is about 2.7 miles in one direction and takes around three hours to complete, according to Parks Canada, which rates the trail as appropriate for moderately experienced hikers.
    """
    # request = """I'm running a test. What's up? (123) My website is https://www.example.com."""

    # Step 2, Step 3 -> Sentence segmentation and sentence embeddings at the sentence level
    document = nlp_model_spaCy(request)
    sentence_data = []
    for index, sentence in enumerate(document.sents):
        
        # Get cleaned sentence
        cleaned_sentence = sentence.text.strip()

        # Get sentence embeddings via tokenization
        sentence_vectorized = [token.vector for token in sentence if (token.has_vector and token.vector.ndim > 0 and not token.is_space and token.text.strip())]
        sentence_embedding = np.zeros(nlp_model_spaCy.vocab.vectors.shape[1], dtype=np.float32)
        if (sentence_vectorized): 
            sentence_embedding = np.mean(sentence_vectorized, axis=0, dtype=np.float32)
        sentence_data.append(
            {
                'sentence': cleaned_sentence,
                'embedding': sentence_embedding,
                'index': index
            }
        )

    if (not sentence_data):
        print('Error: no sentence data collected.')
        return
    
    # Gather all of each type of data
    all_sentence_sentences = np.array([item['sentence'] for item in sentence_data])
    all_sentence_embeddings = np.array([item['embedding'] for item in sentence_data])
    all_sentence_indices = np.array([item['index'] for item in sentence_data])
    num_sentences = len(sentence_data)

    if (not np.any(all_sentence_embeddings)):
        print('Error: all sentence embeddings are zero. Cannot calculate meaningful similarities.')
        
    # Step 4 -> Textrank graph construction and scoring
    sentence_pairwise_similarities = cosine_similarity(all_sentence_embeddings, all_sentence_embeddings)
    non_self_similarities = sentence_pairwise_similarities[~np.eye(num_sentences, dtype=bool)].flatten()
    textrank_edge_threshold = np.percentile(non_self_similarities, 15)
    graph_matrix = np.where(sentence_pairwise_similarities > textrank_edge_threshold, sentence_pairwise_similarities, 0)
    np.fill_diagonal(graph_matrix, 0)
    row_sums = graph_matrix.sum(axis=1)
    graph_matrix = graph_matrix / np.maximum(row_sums, 1e-9)[:, np.newaxis]
    print(graph_matrix.sum(axis=1))
   
    return HttpResponse('dummy')
