import googleapiclient.discovery

import os
import requests
import json
import numpy as np

from preprocessor.preprocess import BertPreprocessor

project=os.environ['GCP_PROJECT_ID']
model="intent_slot"
version='v002'

def intent_slot(request):

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    request_data = request.get_json(silent=False)

    user_question = request_data['user_question']

    instances = []
    instances.append({
        "text_a": user_question, 
        "text_b": None
    })

    preprocessor = BertPreprocessor("vocab.txt", "intents.txt", "slots.txt")

    features = preprocessor.preprocess(instances)

    response = service.projects().predict(
        name=name,
        body={'instances': features}
    ).execute()

    #one item for online prediction
    predictions = response['predictions'][0]

    (intent_labels, slot_labels) = preprocessor.get_labels()

    #one item for online prediction
    words = preprocessor.get_words(instances)[0]

    #one item for online prediction
    tokens = preprocessor.get_tokens(instances)[0]

    intent_probabilities = predictions['intent_probabilities']
    slot_probabilities = predictions['slot_probabilities_padded']

    print ('intent_labels', repr(intent_labels))
    print ('intent_probabilities', repr(intent_probabilities))
    print ('slot_probabilities', repr(slot_probabilities))

    intent = np.argmax(intent_probabilities, axis=0)
    slots = np.argmax(slot_probabilities, axis=1)

    assert len(slots) == len(words)

    slot_names = [slot_labels[i] for i in slots]

    return json.dumps({'intent': intent_labels[intent], 'intent_probability': intent_probabilities[intent], 'slots': slot_names, 'words': words, 'tokens': tokens})

