#!/bin/bash

# Chess Training Visualization Startup Script

echo "🏆 Chess Training Visualizer"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

echo "✅ Python and Node.js found"
echo ""

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd api
if [ ! -f "requirements.txt" ]; then
    echo "❌ api/requirements.txt not found"
    exit 1
fi
pip install -q -r requirements.txt
cd ..
echo "✅ Backend dependencies installed"
echo ""

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
if [ ! -f "package.json" ]; then
    echo "❌ frontend/package.json not found"
    exit 1
fi
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "   (node_modules already exists, skipping)"
fi
cd ..
echo "✅ Frontend dependencies installed"
echo ""

# Start backend in background
echo "🚀 Starting backend API..."
cd api
python server.py &
BACKEND_PID=$!
cd ..
echo "✅ Backend running on http://localhost:8000 (PID: $BACKEND_PID)"
echo ""

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Start frontend in background
echo "🚀 Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo "✅ Frontend running on http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""

echo "=============================="
echo "✨ Visualization dashboard is ready!"
echo ""
echo "📡 Backend API: http://localhost:8000"
echo "🎨 Frontend:    http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=============================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT TERM

# Wait for processes
wait
