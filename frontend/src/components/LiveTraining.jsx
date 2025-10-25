import { useState, useEffect } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import './LiveTraining.css'

function LiveTraining({ ws }) {
  const [game, setGame] = useState(new Chess())
  const [currentMove, setCurrentMove] = useState(0)
  const [gameData, setGameData] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    if (!ws) return

    const handleMessage = (event) => {
      const message = JSON.parse(event.data)
      
      if (message.type === 'move') {
        handleMoveUpdate(message.data)
      } else if (message.type === 'game_complete') {
        handleGameComplete(message.data)
      }
    }

    ws.addEventListener('message', handleMessage)

    return () => {
      ws.removeEventListener('message', handleMessage)
    }
  }, [ws])

  const handleMoveUpdate = (data) => {
    setGameData(data)
    if (data.fen) {
      const newGame = new Chess(data.fen)
      setGame(newGame)
      setCurrentMove(data.move_number || 0)
    }
  }

  const handleGameComplete = (data) => {
    console.log('Game completed:', data)
    // Reset for next game
    setTimeout(() => {
      setGame(new Chess())
      setCurrentMove(0)
      setGameData(null)
    }, 3000)
  }

  return (
    <div className="live-training">
      <div className="live-header">
        <h2>Live Training Board</h2>
        <div className="live-status">
          {isPlaying ? (
            <span className="status-badge playing">🎮 Game in Progress</span>
          ) : (
            <span className="status-badge waiting">⏳ Waiting for next game...</span>
          )}
        </div>
      </div>

      <div className="live-content">
        <div className="board-container">
          <Chessboard 
            position={game.fen()}
            boardWidth={500}
            customBoardStyle={{
              borderRadius: '8px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
            }}
          />
          
          <div className="board-info">
            <div className="info-item">
              <span className="info-label">Move:</span>
              <span className="info-value">{currentMove}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Turn:</span>
              <span className="info-value">{game.turn() === 'w' ? 'White' : 'Black'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Status:</span>
              <span className="info-value">
                {game.isCheckmate() ? '🏁 Checkmate' :
                 game.isStalemate() ? '🤝 Stalemate' :
                 game.isDraw() ? '🤝 Draw' :
                 game.isCheck() ? '⚠️ Check' :
                 '▶️ Playing'}
              </span>
            </div>
          </div>
        </div>

        <div className="game-details">
          <h3>Current Game Details</h3>
          
          {gameData ? (
            <div className="details-content">
              <div className="detail-section">
                <h4>Position Info</h4>
                <div className="detail-item">
                  <span>FEN:</span>
                  <code>{game.fen()}</code>
                </div>
                <div className="detail-item">
                  <span>Legal Moves:</span>
                  <span>{game.moves().length}</span>
                </div>
              </div>

              {gameData.mcts_stats && (
                <div className="detail-section">
                  <h4>MCTS Statistics</h4>
                  <div className="detail-item">
                    <span>Simulations:</span>
                    <span>{gameData.mcts_stats.simulations}</span>
                  </div>
                  <div className="detail-item">
                    <span>Nodes Explored:</span>
                    <span>{gameData.mcts_stats.nodes_explored}</span>
                  </div>
                  <div className="detail-item">
                    <span>Search Time:</span>
                    <span>{gameData.mcts_stats.search_time?.toFixed(2)}s</span>
                  </div>
                </div>
              )}

              {gameData.evaluation && (
                <div className="detail-section">
                  <h4>Neural Network Evaluation</h4>
                  <div className="detail-item">
                    <span>Position Value:</span>
                    <span className={gameData.evaluation.value > 0 ? 'positive' : 'negative'}>
                      {gameData.evaluation.value?.toFixed(3)}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span>Top Move Probability:</span>
                    <span>{(gameData.evaluation.top_move_prob * 100)?.toFixed(1)}%</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="no-data">
              <p>Waiting for training to start...</p>
              <p className="hint">Start training from the Control Panel to see live games</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default LiveTraining
