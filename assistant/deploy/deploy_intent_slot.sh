
gcloud beta ai-platform versions create v002 \
	--model=intent_slot \
	--origin=gs://$GCP_PROJECT_ID.appspot.com/test/intent_slot_output/uncased_L-12_H-768_A-12/deployment/$1/ \
	--runtime-version=1.14 \
	--framework=tensorflow \
	--python-version=3.5 \
	--machine-type=n1-highcpu-2
	#--accelerator=count=1,type=nvidia-tesla-t4 \
	#--accelerator=count=1,type=nvidia-tesla-k80 \
	#--machine-type=n1-standard-2
	#--accelerator=count=1,type=nvidia-tesla-v100 \
	#--machine-type=n1-highcpu-8
