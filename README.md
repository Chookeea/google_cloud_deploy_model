### 1. Write App (Flask, TensorFlow)

### 2. Setup Google Cloud
Create new project
activate cloud run API and cloud builds API

### 3. Install and init Google Cloud SDK
https://cloud.google.com/sdk/docs/install

### 4. Dockerfile, requirements.txt, .dockerignore
https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing

### 5. Cloud build & deploy
gcloud builds submit --tag gcr.io/<project_id>/index
gcloud run deploy --image gcr.io/<project_id>/index --platform managed