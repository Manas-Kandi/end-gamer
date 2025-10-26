#!/usr/bin/env python3
"""
Simple script to run both backend and frontend together.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Check if we're in a virtual environment, if not, try to activate it
if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    venv_path = Path("venv/bin/python")
    if venv_path.exists():
        print("⚠️  Not in virtual environment. Using venv/bin/python")
        # Re-run this script with the venv python
        os.execv(str(venv_path), [str(venv_path)] + sys.argv)
    else:
        print("❌ Virtual environment not found at venv/")
        print("   Please activate your virtual environment first:")
        print("   source venv/bin/activate")
        sys.exit(1)

# Store process references
processes = []

def cleanup(signum=None, frame=None):
    """Cleanup function to kill all processes."""
    print("\n🛑 Stopping all services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("✅ All services stopped")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    print("🏆 Chess Training Visualizer")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("api/server.py").exists():
        print("❌ Error: api/server.py not found")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    if not Path("frontend/package.json").exists():
        print("❌ Error: frontend/package.json not found")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if frontend dependencies are installed
    if not Path("frontend/node_modules").exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd="frontend",
                check=True
            )
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            print("   Please run: cd frontend && npm install")
            sys.exit(1)
    
    print()
    print("🚀 Starting backend API (Demo Mode)...")
    
    # Start backend with demo server
    backend_proc = subprocess.Popen(
        [sys.executable, "api/demo_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(backend_proc)
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Check if backend is still running
    if backend_proc.poll() is not None:
        print("❌ Backend failed to start")
        print("\nBackend output:")
        print(backend_proc.stdout.read())
        cleanup()
        sys.exit(1)
    
    print("✅ Backend running on http://localhost:8000")
    print()
    
    print("🚀 Starting frontend...")
    
    # Start frontend
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(frontend_proc)
    
    # Wait for frontend to start
    time.sleep(3)
    
    # Check if frontend is still running
    if frontend_proc.poll() is not None:
        print("❌ Frontend failed to start")
        print("\nFrontend output:")
        print(frontend_proc.stdout.read())
        cleanup()
        sys.exit(1)
    
    print("✅ Frontend running on http://localhost:4010")
    print()
    print("=" * 50)
    print("✨ Visualization dashboard is ready!")
    print()
    print("📡 Backend API: http://localhost:8000")
    print("🎨 Frontend:    http://localhost:4010")
    print()
    print("Press Ctrl+C to stop all services")
    print("=" * 50)
    print()
    
    # Keep script running and show output
    try:
        while True:
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print("\n❌ Backend process died unexpectedly")
                cleanup()
                sys.exit(1)
            
            if frontend_proc.poll() is not None:
                print("\n❌ Frontend process died unexpectedly")
                cleanup()
                sys.exit(1)
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
