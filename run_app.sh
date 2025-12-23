#!/bin/bash

# Check if .env exists
if [ ! -f backend/.env ]; then
    echo "Creating backend/.env..."
    echo "OPENAI_API_KEY=your_key_here" > backend/.env
fi

echo "Make sure you have set your OPENAI_API_KEY in backend/.env!"
read -p "Press Enter to continue..."

# Install Backend Deps
echo "Installing Backend Dependencies..."
pip install -r backend/requirements.txt

# Start Backend in background
echo "Starting Backend..."
cd backend
python3 server.py &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Backend running at PID $BACKEND_PID"
echo "Frontend running at PID $FRONTEND_PID"
echo "Access the app at http://localhost:3000"

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

wait
