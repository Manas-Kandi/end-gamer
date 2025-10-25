import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './MetricsDashboard.css'

function MetricsDashboard({ metrics }) {
  // Mock historical data - in real implementation, this would come from API
  const mockHistoricalData = [
    { iteration: 0, win_rate: 0.2, draw_rate: 0.4, loss: 2.5 },
    { iteration: 10, win_rate: 0.35, draw_rate: 0.55, loss: 2.1 },
    { iteration: 20, win_rate: 0.48, draw_rate: 0.68, loss: 1.8 },
    { iteration: 30, win_rate: 0.62, draw_rate: 0.75, loss: 1.5 },
    { iteration: 40, win_rate: 0.71, draw_rate: 0.82, loss: 1.2 },
    { iteration: 50, win_rate: 0.79, draw_rate: 0.87, loss: 0.9 },
  ]

  return (
    <div className="metrics-dashboard">
      <div className="dashboard-header">
        <h2>Training Metrics</h2>
      </div>

      <div className="metrics-grid">
        <div className="metric-card large">
          <h3>Win Rate Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockHistoricalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3d3d3d" />
              <XAxis dataKey="iteration" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ background: '#2d2d2d', border: '1px solid #3d3d3d' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="win_rate"
                stroke="#10b981"
                strokeWidth={2}
                name="Win Rate"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="metric-card large">
          <h3>Draw Rate Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockHistoricalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3d3d3d" />
              <XAxis dataKey="iteration" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ background: '#2d2d2d', border: '1px solid #3d3d3d' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="draw_rate"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Draw Rate"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="metric-card large">
          <h3>Training Loss</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockHistoricalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3d3d3d" />
              <XAxis dataKey="iteration" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ background: '#2d2d2d', border: '1px solid #3d3d3d' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#ef4444"
                strokeWidth={2}
                name="Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">🎯</div>
            <div className="metric-info">
              <div className="metric-label">Total Games</div>
              <div className="metric-value">{metrics?.total_games || 0}</div>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">🔄</div>
            <div className="metric-info">
              <div className="metric-label">Current Iteration</div>
              <div className="metric-value">{metrics?.iteration || 0}</div>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">🏆</div>
            <div className="metric-info">
              <div className="metric-label">Win Rate</div>
              <div className="metric-value success">
                {((metrics?.win_rate || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">🤝</div>
            <div className="metric-info">
              <div className="metric-label">Draw Rate</div>
              <div className="metric-value info">
                {((metrics?.draw_rate || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">📉</div>
            <div className="metric-info">
              <div className="metric-label">Current Loss</div>
              <div className="metric-value">{(metrics?.loss || 0).toFixed(4)}</div>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-content">
            <div className="metric-icon">📚</div>
            <div className="metric-info">
              <div className="metric-label">Curriculum Level</div>
              <div className="metric-value">Level {metrics?.curriculum_level || 0}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MetricsDashboard
