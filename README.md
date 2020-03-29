# intent_slot
Intent slot model to use in artificial assistant written in Tensorflow

This is an implementation of "BERT for Joint Intent Classification and Slot Filling" - arXiv:1902.10909v1  [cs.CL]  28 Feb 2019.

There is a joint training using BERT and loss is calculated as total loss from intent and slotss. Objective function is calculated as in equation (3) in the arcticle. 

Data used in this work:

- Snips Voice Platform: an embeddedSpoken Language Understanding systemfor private-by-design voice interfaces - arXiv:1805.10190v3  [cs.CL]  6 Dec 2018 
- An Evaluation Dataset for Intent Classificationand Out-of-Scope Prediction - arXiv:1909.02027v1  [cs.CL]  4 Sep 2019

Snips dataset contain labels for both intent and slots and Out-of-Scope team has data related to financial domain without slot labels and some of this data was labeled manually and merged with Snips.










