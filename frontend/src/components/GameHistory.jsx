import { useState, useEffect } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import './GameHistory.css'

function GameHistory() {
  const [games, setGames] = useState([])
  const [selectedGame, setSelectedGame] = useState(null)
  const [currentPosition, setCurrentPosition] = useState(0)
  const [chess, setChess] = useState(new Chess())
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [totalGames, setTotalGames] = useState(0)

  const GAMES_PER_PAGE = 20

  useEffect(() => {
    fetchGames()
  }, [page])

  const fetchGames = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/games?limit=${GAMES_PER_PAGE}&offset=${page * GAMES_PER_PAGE}`)
      const data = await response.json()
      setGames(data.games)
      setTotalGames(data.total)
    } catch (error) {
      console.error('Error fetching games:', error)
    } finally {
      setLoading(false)
    }
  }

  const selectGame = async (gameId) => {
    try {
      const response = await fetch(`/api/games/${gameId}`)
      const game = await response.json()
      setSelectedGame(game)
      setCurrentPosition(0)
      
      // Initialize chess with starting position
      const newChess = new Chess()
      setChess(newChess)
    } catch (error) {
      console.error('Error fetching game:', error)
    }
  }

  const goToPosition = (positionIndex) => {
    if (!selectedGame) return
    
    const newChess = new Chess()
    
    // Replay moves up to position
    for (let i = 0; i < positionIndex && i < selectedGame.moves.length; i++) {
      newChess.move(selectedGame.moves[i])
    }
    
    setChess(newChess)
    setCurrentPosition(positionIndex)
  }

  const nextMove = () => {
    if (currentPosition < selectedGame?.moves.length) {
      goToPosition(currentPosition + 1)
    }
  }

  const prevMove = () => {
    if (currentPosition > 0) {
      goToPosition(currentPosition - 1)
    }
  }

  const getResultBadge = (result) => {
    if (result === 1) return <span className="result-badge win">Win</span>
    if (result === 0) return <span className="result-badge draw">Draw</span>
    if (result === -1) return <span className="result-badge loss">Loss</span>
    return <span className="result-badge unknown">Unknown</span>
  }

  return (
    <div className="game-history">
      <div className="history-header">
        <h2>Game History</h2>
        <div className="history-stats">
          <span>Total Games: {totalGames}</span>
        </div>
      </div>

      <div className="history-content">
        <div className="games-list">
          <h3>Recent Games</h3>
          
          {loading ? (
            <div className="loading">Loading games...</div>
          ) : games.length === 0 ? (
            <div className="no-games">
              <p>No games yet</p>
              <p className="hint">Start training to generate games</p>
            </div>
          ) : (
            <>
              <div className="games-grid">
                {games.map((game) => (
                  <div
                    key={game.id}
                    className={`game-card ${selectedGame?.id === game.id ? 'selected' : ''}`}
                    onClick={() => selectGame(game.id)}
                  >
                    <div className="game-card-header">
                      <span className="game-id">Game #{game.id}</span>
                      {getResultBadge(game.result)}
                    </div>
                    <div className="game-card-body">
                      <div className="game-stat">
                        <span className="stat-label">Moves:</span>
                        <span className="stat-value">{game.num_moves}</span>
                      </div>
                      <div className="game-stat">
                        <span className="stat-label">Time:</span>
                        <span className="stat-value">
                          {new Date(game.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="pagination">
                <button
                  onClick={() => setPage(Math.max(0, page - 1))}
                  disabled={page === 0}
                  className="pagination-btn"
                >
                  ← Previous
                </button>
                <span className="pagination-info">
                  Page {page + 1} of {Math.ceil(totalGames / GAMES_PER_PAGE)}
                </span>
                <button
                  onClick={() => setPage(page + 1)}
                  disabled={(page + 1) * GAMES_PER_PAGE >= totalGames}
                  className="pagination-btn"
                >
                  Next →
                </button>
              </div>
            </>
          )}
        </div>

        <div className="game-viewer">
          {selectedGame ? (
            <>
              <div className="viewer-header">
                <h3>Game #{selectedGame.id}</h3>
                <div className="viewer-info">
                  {getResultBadge(selectedGame.result)}
                  <span className="move-counter">
                    Move {currentPosition} / {selectedGame.num_moves}
                  </span>
                </div>
              </div>

              <div className="board-wrapper">
                <Chessboard
                  position={chess.fen()}
                  boardWidth={450}
                  customBoardStyle={{
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
                  }}
                />
              </div>

              <div className="viewer-controls">
                <button
                  onClick={() => goToPosition(0)}
                  disabled={currentPosition === 0}
                  className="control-btn"
                >
                  ⏮ Start
                </button>
                <button
                  onClick={prevMove}
                  disabled={currentPosition === 0}
                  className="control-btn"
                >
                  ◀ Prev
                </button>
                <button
                  onClick={nextMove}
                  disabled={currentPosition >= selectedGame.moves.length}
                  className="control-btn"
                >
                  Next ▶
                </button>
                <button
                  onClick={() => goToPosition(selectedGame.moves.length)}
                  disabled={currentPosition >= selectedGame.moves.length}
                  className="control-btn"
                >
                  End ⏭
                </button>
              </div>

              <div className="move-list">
                <h4>Move History</h4>
                <div className="moves-scroll">
                  {selectedGame.moves.map((move, index) => (
                    <div
                      key={index}
                      className={`move-item ${index === currentPosition - 1 ? 'current' : ''}`}
                      onClick={() => goToPosition(index + 1)}
                    >
                      <span className="move-number">{index + 1}.</span>
                      <span className="move-notation">{move}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="no-selection">
              <p>Select a game to view</p>
              <p className="hint">Click on any game card to see its details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default GameHistory
