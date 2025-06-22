pipeline{
    agent any

    environment {
        // Define any environment variables here
        VENV_DIR = 'venv'
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
                    apt install python3.11-venv
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate

                    pip install upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}