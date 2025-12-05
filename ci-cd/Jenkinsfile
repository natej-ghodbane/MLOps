pipeline {
    agent any

    environment {
        DOCKERHUB_CRED = 'dockerhub_cred'
        IMAGE_API = 'noticc/churn-api'
        IMAGE_UI  = 'noticc/churn-ui'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh 'make setup'
            }
        }

        stage('Lint') {
            steps {
                sh 'make lint'
            }
        }

        stage('Format') {
            steps {
                sh 'make format'
            }
        }

        stage('Security') {
            steps {
                sh 'make security'
            }
        }

        stage('Load Data') {
            steps {
                sh 'make load'
            }
        }

        stage('Prepare Data') {
            steps {
                sh 'make prepare'
            }
        }

        stage('Train Model') {
            steps {
                sh 'make train'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'make evaluate'
            }
        }

        stage('Test') {
            steps {
                sh 'make test'
            }
        }

        stage('Docker Build Images') {
            steps {
                sh 'make docker-build'
            }
        }

        stage('Docker Login & Push') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: DOCKERHUB_CRED,
                    usernameVariable: 'USER',
                    passwordVariable: 'PASS'
                )]) {
                    sh 'echo "$PASS" | docker login -u "$USER" --password-stdin'
                }
                sh 'make docker-push'
            }
        }

        stage('Deploy App with Docker Compose') {
            steps {
                sh 'make docker-redeploy'
            }
        }
    }
}
