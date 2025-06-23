pipeline{
    agent any

    environment {
        // Define any environment variables here
        VENV_DIR = 'venv'
        GCP_PROJECT = "eastern-period-463504-e2"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk"
    }

    stages {
        stage('Cloning Github repo to jenkins') {
            steps {
                script {
                    echo 'Cloning the repository...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Nevin-A-S/MLOPS_Hotel_Reservation.git']])
                }
            }
        }

        stage('setting up virtual environment and installing dependencies') {
            steps {
                script {
                    echo 'setting up virtual environment and installing dependencies...'
                    sh'''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate

                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Train Model') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo 'Authenticating to GCP and running training pipeline...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        # Activate the virtual environment and run the training script
                        echo "Running the training script..."
                        . ${VENV_DIR}/bin/activate
                        python pipeline/training_pipeline.py
                        '''
                    }
                }
            }
        }

        stage('Builing and Pushing Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo 'Building and pushing Docker image to GCR...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/mlops-hotel-reservation:latest .

                        docker push gcr.io/${GCP_PROJECT}/mlops-hotel-reservation:latest

                        '''
                    }
                }
            }
        }
    }
}