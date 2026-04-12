import { useState, useEffect } from 'react';
import { RefreshCw, Play, Package, TrendingUp, AlertTriangle, AlertCircle, Calendar, Truck, Database, BarChart3 } from 'lucide-react';
import './index.css';

const API_BASE = '';

// Task descriptions for scenario presets
const TASK_PRESETS = {
  task1_single_product: {
    label: 'Easy — Single Product',
    desc: '1 non-perishable product, stable demand, fixed lead time. Learn basic reorder policy.',
  },
  task2_multi_product: {
    label: 'Medium — Multi-Product',
    desc: '3 products (1 perishable), seasonal demand, shared storage. Balance waste vs stockouts.',
  },
  task3_nonstationary: {
    label: 'Hard — Non-Stationary',
    desc: '5 products, demand shocks, unreliable suppliers, emergency orders. Full complexity.',
  },
};

function App() {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState('task1_single_product');
  const [stateData, setStateData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [actionIds, setActionIds] = useState([]);
  const [lastInfo, setLastInfo] = useState(null);
  const [score, setScore] = useState(null);
  const [legalActions, setLegalActions] = useState([]);
  const [productNames, setProductNames] = useState([]);
  const [stepCount, setStepCount] = useState(0);
  const [maxSteps, setMaxSteps] = useState(0);
  const [episodeRewards, setEpisodeRewards] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/tasks`)
      .then(res => res.json())
      .then(data => setTasks(data))
      .catch(err => {
        console.error("Failed to fetch tasks", err);
        setError("Failed to connect to API server. Ensure it is running on port 7860.");
      });
  }, []);

  const applyResetData = (data) => {
    setStateData(data.state);
    setProductNames(data.product_names || []);
    setLegalActions(data.legal_actions || []);
    setMaxSteps(data.max_steps || 0);
    setStepCount(0);
    setEpisodeRewards([]);
    setActionIds(new Array(data.num_products).fill(0));
  };

  const handleReset = async () => {
    setLoading(true);
    setError(null);
    setScore(null);
    setLastInfo(null);
    try {
      const res = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: selectedTask })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      applyResetData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStep = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action_ids: actionIds })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStateData(data.state);
      setLastInfo(data.info);
      setStepCount(prev => prev + 1);
      setEpisodeRewards(data.episode_rewards || []);
      if (data.done) {
        setScore(data.score);
      }
      setActionIds(new Array(data.state.inventory.length).fill(0));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (tasks.length > 0 && !stateData && !loading) {
      handleReset();
    }
  }, [tasks, selectedTask]);

  const handleActionChange = (productIdx, actionIndex) => {
    const updated = [...actionIds];
    updated[productIdx] = parseInt(actionIndex, 10);
    setActionIds(updated);
  };

  // Error state
  if (error) {
    return (
      <div className="dashboard-container">
        <div className="glass-panel error-state">
          <AlertCircle size={48} />
          <h2>Connection Error</h2>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            <RefreshCw size={18} /> Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!stateData) {
    return (
      <div className="dashboard-container">
        <div className="glass-panel loading-state">
          <RefreshCw className="animate-spin" size={48} color="var(--accent-blue)" />
          <h2>Initializing Environment...</h2>
        </div>
      </div>
    );
  }

  const { inventory, in_transit, days_to_expiry, demand_history, storage_used, day_of_week } = stateData;
  const numProducts = inventory.length;
  const storageUsedValue = typeof storage_used === 'number' ? storage_used : (Array.isArray(storage_used) ? storage_used[0] : 0);
  const getDayName = (dow) => ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow] || "Day";
  const preset = TASK_PRESETS[selectedTask] || {};

  return (
    <div className="dashboard-container fade-enter">
      <header className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Warehouse Control</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Adaptive Multi-Product Inventory Management Agent
          </p>
        </div>
        
        <div className="controls-group">
          <select 
            className="task-select" 
            value={selectedTask}
            onChange={(e) => setSelectedTask(e.target.value)}
            id="task-selector"
          >
            {tasks.map(t => (
              <option key={t.id} value={t.id}>
                {TASK_PRESETS[t.id]?.label || t.id}
              </option>
            ))}
          </select>

          <button className="btn btn-danger" onClick={handleReset} disabled={loading} id="reset-btn">
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
            Reset
          </button>
        </div>
      </header>

      {/* Scenario Description */}
      {preset.desc && (
        <div className="scenario-banner">
          <AlertTriangle size={16} color="var(--accent-orange)" />
          <span>{preset.desc}</span>
        </div>
      )}

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-item">
          <Calendar size={18} color="var(--accent-purple)" />
          <span>Day: <strong>{getDayName(day_of_week)}</strong></span>
        </div>
        <div className="status-item">
          <Play size={18} color="var(--accent-blue)" />
          <span>Step: <strong>{stepCount} / {maxSteps}</strong></span>
        </div>
        <div className="status-item" style={{ flexGrow: 1 }}>
          <Database size={18} color="var(--accent-orange)" />
          <div style={{ flexGrow: 1 }}>
            <div className="progress-label">
              <span>Storage Capacity</span>
              <span>{(storageUsedValue * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-track" style={{ height: '6px' }}>
              <div 
                className={`progress-fill ${storageUsedValue > 0.9 ? 'bg-red' : storageUsedValue > 0.7 ? 'bg-orange' : 'bg-blue'}`} 
                style={{ width: `${Math.min(storageUsedValue * 100, 100)}%` }} 
              />
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid">
        <div className="glass-panel metric-card">
          <div className="metric-header"><TrendingUp size={16} /> Total Revenue</div>
          <div className="metric-value positive">${lastInfo ? lastInfo?.total_revenue?.toFixed(0) : "0"}</div>
          <span className="metric-sub">
            Step: ${lastInfo?.step_revenue?.toFixed(0) || "0"}
          </span>
        </div>

        <div className="glass-panel metric-card">
          <div className="metric-header"><Package size={16} /> Fill Rate</div>
          <div className={`metric-value ${(lastInfo?.fill_rate || 0) > 0.9 ? 'positive' : (lastInfo?.fill_rate > 0.6 ? 'neutral' : 'negative')}`}>
            {lastInfo ? (lastInfo.fill_rate * 100).toFixed(1) : "0.0"}%
          </div>
          <span className="metric-sub">
            Step: {lastInfo ? ((lastInfo.step_fill_rate || 0) * 100).toFixed(1) : "0.0"}%
          </span>
        </div>

        <div className="glass-panel metric-card">
          <div className="metric-header"><AlertTriangle size={16} /> Costs / Penalties</div>
          <div className="metric-value negative">
            ${lastInfo ? ((lastInfo.total_holding_cost || 0) + (lastInfo.total_ordering_cost || 0)).toFixed(0) : "0"}
          </div>
          <span className="metric-sub">
            Waste: {lastInfo ? ((lastInfo.waste_rate || 0) * 100).toFixed(1) : "0.0"}%
          </span>
        </div>

        {lastInfo && lastInfo.service_level !== undefined && (
          <div className="glass-panel metric-card">
            <div className="metric-header"><BarChart3 size={16} /> Service Level</div>
            <div className={`metric-value ${(lastInfo.service_level || 0) > 0.8 ? 'positive' : 'neutral'}`}>
              {((lastInfo.service_level || 0) * 100).toFixed(1)}%
            </div>
            <span className="metric-sub">
              Days with ≥95% fill rate
            </span>
          </div>
        )}

        {score !== null && (
          <div className="glass-panel metric-card score-card">
            <div className="metric-header" style={{color: 'var(--accent-green)'}}>Final Score</div>
            <div className="metric-value positive">{(score * 100).toFixed(1)}</div>
            <span className="metric-sub">Episode Complete!</span>
          </div>
        )}
      </div>

      {/* Reward History Sparkline */}
      {episodeRewards.length > 1 && (
        <div className="glass-panel chart-panel">
          <div className="chart-header">
            <BarChart3 size={16} />
            <span>Episode Reward History</span>
            <span className="chart-stat">
              Avg: {(episodeRewards.reduce((a,b)=>a+b,0) / episodeRewards.length).toFixed(3)}
            </span>
          </div>
          <RewardSparkline data={episodeRewards} />
        </div>
      )}

      {/* Products Grid */}
      <h2 className="section-title"><Package /> Product Status & Orders</h2>
      <div className="products-grid">
        {inventory.map((inv, idx) => {
          const isPerishable = days_to_expiry[idx] !== -1;
          const transitArray = in_transit[idx] || [];
          const incoming = transitArray.reduce((a,b) => a+b, 0);
          const name = productNames[idx] || `Product ${idx + 1}`;
          
          // Legal actions come exclusively from API — single source of truth
          const productLegalActions = legalActions[idx]?.legal_actions || [];

          return (
            <div key={idx} className="glass-panel product-card">
              <div className="product-header">
                <div className="product-title">{name}</div>
                <div className="product-badges">
                  {isPerishable && <span className="badge badge-perishable">Perishable</span>}
                </div>
              </div>

              <div className="progress-group">
                <div className="progress-label">
                  <span>Inventory Level</span>
                  <strong>{inv.toFixed(0)} units</strong>
                </div>
                <div className="progress-track">
                  <div className={`progress-fill ${inv < 20 ? 'bg-red' : inv > 150 ? 'bg-orange' : 'bg-green'}`} style={{ width: `${Math.min((inv / 200) * 100, 100)}%` }} />
                </div>
              </div>

              {isPerishable && (
                <div style={{ fontSize: '0.85rem', color: days_to_expiry[idx] <= 2 ? 'var(--accent-red)' : 'var(--text-secondary)', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <AlertCircle size={14} /> Min days to expiry: <strong>{days_to_expiry[idx]}</strong>
                </div>
              )}

              {incoming > 0 && (
                <div className="in-transit-container">
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', width: '100%' }}>In Transit:</div>
                  {transitArray.map((qty, d) => qty > 0 ? (
                    <span key={d} className="in-transit-pill">
                      <Truck size={12} /> {qty.toFixed(0)} in {d + 1}d
                    </span>
                  ) : null)}
                </div>
              )}

              {demand_history && demand_history[idx] && (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Avg demand (7d): <strong>{(demand_history[idx].reduce((a,b) => a+b, 0) / Math.max(demand_history[idx].filter(d => d > 0).length, 1)).toFixed(1)}</strong> units/day
                </div>
              )}

              <div className="order-controls">
                <div className="order-label">
                  Reorder Action <span style={{ color: 'var(--accent-blue)', fontSize: '0.75rem' }}>(index {actionIds[idx] || 0})</span>
                </div>
                <div className="action-selector">
                  {productLegalActions.map((action) => (
                    <button
                      key={action.index}
                      className={`action-btn ${(actionIds[idx] || 0) === action.index ? 'action-btn-active' : ''} ${action.is_emergency ? 'action-btn-emergency' : ''}`}
                      onClick={() => handleActionChange(idx, action.index)}
                      disabled={score !== null}
                      title={`Action index ${action.index}: ${action.label}`}
                      id={`action-btn-${idx}-${action.index}`}
                    >
                      {action.is_emergency ? '🚨 ' : ''}{action.label}
                    </button>
                  ))}
                </div>
              </div>

            </div>
          );
        })}
      </div>

      <div className="main-actions">
        <button 
          className="btn btn-primary btn-large" 
          onClick={handleStep}
          disabled={loading || score !== null}
          id="step-btn"
        >
          {score !== null ? "Episode Done — Reset to Continue" : "Execute Step & Advance Day"} <Play size={20} />
        </button>
      </div>

    </div>
  );
}

/** SVG sparkline for episode reward history */
function RewardSparkline({ data }) {
  if (!data || data.length < 2) return null;

  const width = 800;
  const height = 80;
  const padding = 4;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((v - min) / range) * (height - 2 * padding);
    return `${x},${y}`;
  });

  const areaPoints = [
    `${padding},${height - padding}`,
    ...points,
    `${width - padding},${height - padding}`,
  ];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="sparkline-svg" preserveAspectRatio="none">
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent-blue)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="var(--accent-blue)" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon points={areaPoints.join(' ')} fill="url(#sparkGrad)" />
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke="var(--accent-blue)"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default App;
