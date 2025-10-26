import { useState } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import './PositionTraining.css'

function PositionTraining() {
  const [game, setGame] = useState(new Chess())
  const [fen, setFen] = useState(game.fen())
  const [savedPositions, setSavedPositions] = useState([])
  const [positionName, setPositionName] = useState('')
  const [isTraining, setIsTraining] = useState(false)

  const onDrop = (sourceSquare, targetSquare) => {
    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      })

      if (move === null) return false

      setFen(game.fen())
      return true
    } catch {
      return false
    }
  }

  const resetPosition = () => {
    const newGame = new Chess()
    setGame(newGame)
    setFen(newGame.fen())
  }

  const clearBoard = () => {
    const newGame = new Chess('8/8/8/8/8/8/8/8 w - - 0 1')
    setGame(newGame)
    setFen(newGame.fen())
  }

  const savePosition = () => {
    if (!positionName.trim()) return

    const position = {
      id: Date.now(),
      name: positionName,
      fen: fen,
      timestamp: new Date().toISOString()
    }

    setSavedPositions([...savedPositions, position])
    setPositionName('')
  }

  const loadPosition = (savedFen) => {
    const newGame = new Chess(savedFen)
    setGame(newGame)
    setFen(savedFen)
  }

  const deletePosition = (id) => {
    setSavedPositions(savedPositions.filter(p => p.id !== id))
  }

  const startPositionTraining = async () => {
    setIsTraining(true)
    
    try {
      const response = await fetch('/api/training/position/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: fen })
      })
      
      if (response.ok) {
        console.log('Position training started')
      }
    } catch (error) {
      console.error('Failed to start position training:', error)
    }
    
    setIsTraining(false)
  }

  return (
    <div className="position-training">
      <div className="position-header">
        <h2>Position Training</h2>
        <p>Set up a position and train the model specifically for it</p>
      </div>

      <div className="position-content">
        <div className="board-section">
          <div className="board-wrapper">
            <Chessboard
              position={fen}
              onPieceDrop={onDrop}
              boardWidth={480}
              customBoardStyle={{
                borderRadius: '2px',
              }}
              customDarkSquareStyle={{ backgroundColor: '#2a2a2a' }}
              customLightSquareStyle={{ backgroundColor: '#3a3a3a' }}
            />
          </div>

          <div className="board-controls">
            <button onClick={resetPosition} className="control-btn">
              Reset
            </button>
            <button onClick={clearBoard} className="control-btn">
              Clear
            </button>
            <button 
              onClick={startPositionTraining} 
              className="control-btn primary"
              disabled={isTraining}
            >
              {isTraining ? 'Training...' : 'Train Position'}
            </button>
          </div>

          <div className="fen-display">
            <label>FEN</label>
            <input 
              type="text" 
              value={fen} 
              onChange={(e) => {
                try {
                  const newGame = new Chess(e.target.value)
                  setGame(newGame)
                  setFen(e.target.value)
                } catch {}
              }}
              className="fen-input"
            />
          </div>
        </div>

        <div className="positions-section">
          <div className="save-position">
            <h3>Save Position</h3>
            <input
              type="text"
              placeholder="Position name"
              value={positionName}
              onChange={(e) => setPositionName(e.target.value)}
              className="position-name-input"
            />
            <button onClick={savePosition} className="save-btn">
              Save
            </button>
          </div>

          <div className="saved-positions">
            <h3>Saved Positions</h3>
            {savedPositions.length === 0 ? (
              <p className="empty-message">No saved positions</p>
            ) : (
              <div className="positions-list">
                {savedPositions.map((pos) => (
                  <div key={pos.id} className="position-item">
                    <div className="position-info">
                      <span className="position-name">{pos.name}</span>
                      <span className="position-date">
                        {new Date(pos.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="position-actions">
                      <button
                        onClick={() => loadPosition(pos.fen)}
                        className="action-btn"
                      >
                        Load
                      </button>
                      <button
                        onClick={() => deletePosition(pos.id)}
                        className="action-btn delete"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PositionTraining
