#!/usr/bin/env python3
"""Quick test to see if backend can start."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing backend imports...")

try:
    from src.config.config import Config
    print("✅ Config imported")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

try:
    from src.chess_env.position import Position
    print("✅ Position imported")
except Exception as e:
    print(f"❌ Position import failed: {e}")
    sys.exit(1)

try:
    import chess
    print("✅ chess imported")
except Exception as e:
    print(f"❌ chess import failed: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nNow testing FastAPI...")

try:
    from fastapi import FastAPI
    print("✅ FastAPI imported")
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"status": "ok"}
    
    print("✅ FastAPI app created")
    print("\nStarting server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
except Exception as e:
    print(f"❌ FastAPI test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
