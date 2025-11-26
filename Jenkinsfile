pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Setup') {
            steps { sh 'make setup' }
        }
        stage('Lint') {
            steps { sh 'make lint' }
        }
        stage('Format') {
            steps { sh 'make format' }
        }
        stage('Security') {
            steps { sh 'make security' }
        }
        stage('Load Data') {
            steps { sh 'make load' }
        }
        stage('Prepare Data') {
            steps { sh 'make prepare' }
        }
        stage('Train Model') {
            steps { sh 'make train' }
        }
        stage('Evaluate Model') {
            steps { sh 'make evaluate' }
        }
        stage('Test') {
            steps { sh 'make test' }
        }
    }
}
