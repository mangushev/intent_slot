
gcloud functions deploy $1 --allow-unauthenticated --trigger-http --timeout=240 --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS,GCP_PROJECT_ID=$GCP_PROJECT_ID
