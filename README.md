Intent slot model used in artificial assistant (written in Tensorflow)

This is an implementation of "BERT for Joint Intent Classification and Slot Filling" - arXiv:1902.10909v1  [cs.CL]  28 Feb 2019.

Joint training using BERT and loss is calculated as total loss from intent and slots. Objective function is calculated as in equation (3) in the arcticle. 

Please make yourself familiar with BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding - arXiv:1810.04805v2  [cs.CL]  24 May 2019

BERT is extended to train model for joint intent and slots.

Data used in this work:

- Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces - arXiv:1805.10190v3  [cs.CL]  6 Dec 2018 
- An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction - arXiv:1909.02027v1  [cs.CL]  4 Sep 2019

Snips dataset contains labels for both intent and slots and "Out-of-Scope Prediction" has data related to financial domain without slot labels and some of this data was labeled manually and merged with Snips.

Contains of the repository:

- run_intent_slot.py script in bert forder is used the same way as BERT's run_classifier.py

- data folder: train.tsv, dev.tsv, test.tsv. All of them have sentence, intent_label, slot labels separated by tab. But sentence tokens and slot label tokens are separated by space. 

Intents and slot labels came from Snips for: 
AddToPlaylist
BookRestaurant
GetWeather
PlayMusic
RateBook
SearchCreativeWork
SearchScreeningEvent

Slot are manually labeled for:
bill_due
report_fraud
transfer

- train folder has three scripts:INTENT_SLOT.deployment, INTENT_SLOT.evaluate, INTENT_SLOT.predict

evaluate is used for both training and evaluation using train.tsv and dev.tsv data files.
predict uses test.tsv
deployment creates saved_model.pb file and variables which are used to deploy model is a way so it can be used for online prediction

- assistant/deploy/deploy_intent_slot.sh deploys .pb model with variables to gcp ai platform
- assistant/functions/intent_slot contains function that is fronting model and provides better interface for consuming application
- assistant/functions/deploy_function.sh deploys function to gcloud 
- assistant/appengine/intent_slot contains sample application to invoke function to run intent_slot prediction

Steps:

1. Create gcp project, create any VM instance, use standard 2-4 cpu, debian 9 deep learning TF 1.15. Create TPU instance, TF 1.15, preemptive (but keep it off until you start training!)
2. Get BERT and it should be in the bert folder. Put run_intent_slot.py into this folder as well
3. Get gcp storage, create test folders test/pretrained/uncased_L-12_H-768_A-12/ and put unzipped BERT pretrained uncased_L-12_H-768_A-12 model content there. Use gutil cp
4. Set varibles GCP_PROJECT_ID as your gcp project is, TPU_NAME as name of your tpu instance, INTENT_SLOT_FOLDER to the root folder of this software
5. Start TPU. Go to train folder and run: sh INTENT_SLOT.evaluate
