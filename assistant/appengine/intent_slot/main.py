from flask import Flask, render_template, request

import os
import json
import requests
import time
import numpy as np
import logging

app = Flask(__name__)

headers = {
    "content-type": "application/json"
}

intent_url=os.environ['INTENT_SLOT_URL']

confidence_threshold = 0.0

def intent_slot_predict(user_question):

    intent_data = json.dumps(
        {
            "user_question": user_question
            
        }
    )   

    print (intent_url)

    intent_json_response = requests.post(intent_url, data=intent_data, headers=headers)

    print (repr(intent_json_response))

    predictions = json.loads(intent_json_response.text)

    return predictions


@app.route('/')
def index():
    return render_template(
        'index.html',
        user_question=None,
        predicted_intent=None,
        predicted_slots=None,
        words=None,
        tokens=None,
        time=None,
        error=None)


@app.route("/result" , methods=['GET', 'POST'])
def result():

    error = None

    t=time.time()

    user_question = request.form.get('user_question')

    predictions = intent_slot_predict(user_question)

    return render_template(
        'index.html',
        user_question=user_question,
        predicted_intent=predictions['intent'],
        predicted_slots=predictions['slots'],
        words=predictions['words'],
        tokens=predictions['tokens'],
        elapsed_time=time.time()-t,
        error=error)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
