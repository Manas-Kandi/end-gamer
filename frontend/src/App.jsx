import { useState, useEffect } from 'react'
import LiveTraining from './components/LiveTraining'
import GameHistory from './components/GameHistory'
import MetricsDashboard from './components/MetricsDashboard'
import ControlPanel from './components/ControlPanel'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [ws, setWs] = useState(null)

  useEffect(() => {
    // Fetch initial status
    fetchStatus()

    // Setup WebSocket connection
    const websocket = new WebSocket('ws://localhost:8000/ws')
    
    websocket.onopen = () => {
      console.log('WebSocket connected')
      setWs(websocket)
    }

    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data)
      handleWebSocketMessage(message)
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    websocket.onclose = () => {
      console.log('WebSocket disconnected')
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        window.location.reload()
      }, 3000)
    }

    return () => {
      if (websocket) {
        websocket.close()
      }
    }
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/status')
      const data = await response.json()
      setTrainingStatus(data)
      setMetrics(data.metrics)
    } catch (error) {
      console.error('Error fetching status:', error)
    }
  }

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'metrics':
        setMetrics(message.data)
        break
      case 'move':
        // Handle live move update
        break
      case 'game_complete':
        // Handle game completion
        break
      default:
        console.log('Unknown message type:', message.type)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏆 Chess Training Visualizer</h1>
        <div className="status-indicator">
          <span className={`status-dot ${trainingStatus?.is_training ? 'active' : 'inactive'}`}></span>
          <span>{trainingStatus?.is_training ? 'Training Active' : 'Idle'}</span>
        </div>
      </header>

      <div className="app-container">
        <aside className="sidebar">
          <nav className="nav-tabs">
            <button 
              className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              📡 Live Training
            </button>
            <button 
              className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              📚 Game History
            </button>
            <button 
              className={`nav-tab ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              📊 Metrics
            </button>
            <button 
              className={`nav-tab ${activeTab === 'control' ? 'active' : ''}`}
              onClick={() => setActiveTab('control')}
            >
              ⚙️ Control Panel
            </button>
          </nav>

          {metrics && (
            <div className="sidebar-metrics">
              <h3>Quick Stats</h3>
              <div className="metric-item">
                <span className="metric-label">Total Games:</span>
                <span className="metric-value">{metrics.total_games}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Iteration:</span>
                <span className="metric-value">{metrics.iteration}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Win Rate:</span>
                <span className="metric-value">{(metrics.win_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Draw Rate:</span>
                <span className="metric-value">{(metrics.draw_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Loss:</span>
                <span className="metric-value">{metrics.loss.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Curriculum:</span>
                <span className="metric-value">Level {metrics.curriculum_level}</span>
              </div>
            </div>
          )}
        </aside>

        <main className="main-content">
          {activeTab === 'live' && <LiveTraining ws={ws} />}
          {activeTab === 'history' && <GameHistory />}
          {activeTab === 'metrics' && <MetricsDashboard metrics={metrics} />}
          {activeTab === 'control' && <ControlPanel onStatusChange={fetchStatus} />}
        </main>
      </div>
    </div>
  )
}

export default App
