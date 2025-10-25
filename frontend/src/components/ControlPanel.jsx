import { useState } from 'react'
import './ControlPanel.css'

function ControlPanel({ onStatusChange }) {
  const [configPath, setConfigPath] = useState('configs/default.yaml')
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [message, setMessage] = useState(null)

  const startTraining = async () => {
    setIsStarting(true)
    setMessage(null)
    
    try {
      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: configPath })
      })
      
      if (response.ok) {
        setMessage({ type: 'success', text: 'Training started successfully!' })
        onStatusChange()
      } else {
        const error = await response.json()
        setMessage({ type: 'error', text: error.detail || 'Failed to start training' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Network error: ' + error.message })
    } finally {
      setIsStarting(false)
    }
  }

  const stopTraining = async () => {
    setIsStopping(true)
    setMessage(null)
    
    try {
      const response = await fetch('/api/training/stop', {
        method: 'POST'
      })
      
      if (response.ok) {
        setMessage({ type: 'success', text: 'Training stopped successfully!' })
        onStatusChange()
      } else {
        const error = await response.json()
        setMessage({ type: 'error', text: error.detail || 'Failed to stop training' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Network error: ' + error.message })
    } finally {
      setIsStopping(false)
    }
  }

  return (
    <div className="control-panel">
      <div className="panel-header">
        <h2>Training Control Panel</h2>
        <p className="panel-description">
          Start, stop, and configure training sessions
        </p>
      </div>

      <div className="panel-content">
        <div className="control-section">
          <h3>Configuration</h3>
          <div className="form-group">
            <label htmlFor="config-path">Config File Path</label>
            <select
              id="config-path"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              className="form-select"
            >
              <option value="configs/default.yaml">Default Configuration</option>
              <option value="configs/quick_test.yaml">Quick Test (Fast)</option>
              <option value="configs/full_training.yaml">Full Training (100K games)</option>
            </select>
            <p className="form-hint">
              Select a configuration file for training parameters
            </p>
          </div>
        </div>

        <div className="control-section">
          <h3>Training Controls</h3>
          <div className="button-group">
            <button
              onClick={startTraining}
              disabled={isStarting}
              className="control-button primary"
            >
              {isStarting ? '⏳ Starting...' : '▶️ Start Training'}
            </button>
            <button
              onClick={stopTraining}
              disabled={isStopping}
              className="control-button danger"
            >
              {isStopping ? '⏳ Stopping...' : '⏹️ Stop Training'}
            </button>
          </div>

          {message && (
            <div className={`message ${message.type}`}>
              {message.type === 'success' ? '✅' : '❌'} {message.text}
            </div>
          )}
        </div>

        <div className="control-section">
          <h3>Configuration Presets</h3>
          <div className="presets-grid">
            <div className="preset-card">
              <div className="preset-header">
                <h4>🚀 Quick Test</h4>
                <span className="preset-badge">Fast</span>
              </div>
              <ul className="preset-details">
                <li>1,000 games</li>
                <li>200 MCTS simulations</li>
                <li>2 workers</li>
                <li>~30 minutes</li>
              </ul>
            </div>

            <div className="preset-card">
              <div className="preset-header">
                <h4>⚡ Default</h4>
                <span className="preset-badge">Balanced</span>
              </div>
              <ul className="preset-details">
                <li>10,000 games</li>
                <li>400 MCTS simulations</li>
                <li>4 workers</li>
                <li>~5 hours</li>
              </ul>
            </div>

            <div className="preset-card">
              <div className="preset-header">
                <h4>🎯 Full Training</h4>
                <span className="preset-badge">Complete</span>
              </div>
              <ul className="preset-details">
                <li>100,000 games</li>
                <li>400 MCTS simulations</li>
                <li>8 workers</li>
                <li>~48 hours</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="control-section">
          <h3>System Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">API Status:</span>
              <span className="info-value success">🟢 Connected</span>
            </div>
            <div className="info-item">
              <span className="info-label">WebSocket:</span>
              <span className="info-value success">🟢 Active</span>
            </div>
            <div className="info-item">
              <span className="info-label">Backend:</span>
              <span className="info-value">FastAPI</span>
            </div>
            <div className="info-item">
              <span className="info-label">Frontend:</span>
              <span className="info-value">React + Vite</span>
            </div>
          </div>
        </div>

        <div className="control-section">
          <h3>Quick Actions</h3>
          <div className="quick-actions">
            <button className="action-button">
              📊 View Logs
            </button>
            <button className="action-button">
              💾 Export Data
            </button>
            <button className="action-button">
              🔄 Reload Config
            </button>
            <button className="action-button">
              🗑️ Clear History
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ControlPanel
