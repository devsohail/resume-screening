#!/bin/bash

# Setup script for Resume Screening System
set -e

echo "Setting up Resume Screening System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Initialize database
echo "Would you like to initialize the database? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Initializing database..."
    python -c "from backend.storage.db_handler import get_db_handler; get_db_handler().create_tables()"
fi

echo "Setup complete!"
echo "To start the backend: cd backend && uvicorn backend.api.main:app --reload"
echo "To start the frontend: cd frontend && npm run dev"

