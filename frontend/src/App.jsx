import { useState, useEffect } from 'react';
import { RefreshCw, Play, Package, TrendingUp, AlertTriangle, AlertCircle, Calendar, Truck, Database } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState('task1_single_product');
  const [stateData, setStateData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [orderQuantities, setOrderQuantities] = useState([]);
  const [lastInfo, setLastInfo] = useState(null);
  const [score, setScore] = useState(null);

  // Fetch available tasks on mount
  useEffect(() => {
    fetch(`${API_BASE}/tasks`)
      .then(res => res.json())
      .then(data => setTasks(data))
      .catch(err => {
        console.error("Failed to fetch tasks", err);
        setError("Failed to connect to API server. Ensure it is running on port 8000.");
      });
  }, []);

  // Set environment state and sync inputs
  const applyState = (data) => {
    setStateData(data.state);
    if (orderQuantities.length !== data.num_products && data.state.inventory) {
      setOrderQuantities(new Array(data.state.inventory.length).fill(0));
    }
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
      applyState(data);
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
        body: JSON.stringify({ action: orderQuantities })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStateData(data.state);
      setLastInfo(data.info);
      if (data.done) {
        setScore(data.score);
      }
      // Reset sliders after ordering
      setOrderQuantities(new Array(data.state.inventory.length).fill(0));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Initial reset automatically
  useEffect(() => {
    if (tasks.length > 0 && !stateData && !loading) {
      handleReset();
    }
  }, [tasks, selectedTask]);

  const handleOrderChange = (idx, val) => {
    const newQ = [...orderQuantities];
    newQ[idx] = parseInt(val, 10);
    setOrderQuantities(newQ);
  };

  // Safe checks
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

  const getDayName = (dow) => ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow] || "Day";

  return (
    <div className="dashboard-container fade-enter">
      <header className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Warehouse Control</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Advanced Agentic Inventory Management</p>
        </div>
        
        <div className="controls-group">
          <select 
            className="task-select" 
            value={selectedTask}
            onChange={(e) => {
              setSelectedTask(e.target.value);
              // Trigger reset on change is handled by useEffect if we wanted to, 
              // but explicit reset is better.
            }}
          >
            {tasks.map(t => (
              <option key={t.id} value={t.id}>{t.difficulty.toUpperCase()}: {t.id}</option>
            ))}
          </select>

          <button className="btn btn-danger" onClick={handleReset} disabled={loading}>
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
            Reset
          </button>
        </div>
      </header>

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-item">
          <Calendar size={18} color="var(--accent-purple)" />
          <span>Day: <strong>{getDayName(day_of_week)}</strong></span>
        </div>
        <div className="status-item" style={{ flexGrow: 1 }}>
          <Database size={18} color="var(--accent-orange)" />
          <div style={{ flexGrow: 1 }}>
            <div className="progress-label">
              <span>Overall Storage Capacity</span>
              <span>{(storage_used * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-track" style={{ height: '6px' }}>
              <div 
                className={`progress-fill ${storage_used > 0.9 ? 'bg-red' : storage_used > 0.7 ? 'bg-orange' : 'bg-blue'}`} 
                style={{ width: `${Math.min(storage_used * 100, 100)}%` }} 
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
          <span style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
            Step: ${lastInfo?.step_revenue?.toFixed(0) || "0"}
          </span>
        </div>

        <div className="glass-panel metric-card">
          <div className="metric-header"><Package size={16} /> Fill Rate</div>
          <div className={`metric-value ${(lastInfo?.fill_rate || 0) > 0.9 ? 'positive' : (lastInfo?.fill_rate > 0.6 ? 'neutral' : 'negative')}`}>
            {lastInfo ? (lastInfo.fill_rate * 100).toFixed(1) : "0.0"}%
          </div>
          <span style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
            Step: {lastInfo ? (lastInfo.step_fill_rate * 100).toFixed(1) : "0.0"}%
          </span>
        </div>

        <div className="glass-panel metric-card">
          <div className="metric-header"><AlertTriangle size={16} /> Costs / Penalties</div>
          <div className="metric-value negative">
            ${lastInfo ? (lastInfo.total_holding_cost + lastInfo.total_ordering_cost + lastInfo.step_stockout_penalty).toFixed(0) : "0"}
          </div>
          <span style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>
             Waste: {lastInfo ? (lastInfo.waste_rate * 100).toFixed(1) : "0.0"}%
          </span>
        </div>

        {score !== null && (
          <div className="glass-panel metric-card" style={{ boxShadow: '0 0 20px rgba(16, 185, 129, 0.2)', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
            <div className="metric-header" style={{color: 'var(--accent-green)'}}>Final Score</div>
            <div className="metric-value positive">{(score * 100).toFixed(1)}</div>
            <span style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Episode Complete!</span>
          </div>
        )}
      </div>

      {/* Products Grid */}
      <h2 className="section-title"><Package /> Product Status & Orders</h2>
      <div className="products-grid">
        {inventory.map((inv, idx) => {
          const isPerishable = days_to_expiry[idx] !== -1;
          const transitArray = in_transit[idx] || [];
          const incoming = transitArray.reduce((a,b) => a+b, 0);
          
          return (
            <div key={idx} className="glass-panel product-card">
              <div className="product-header">
                <div className="product-title">Product {idx + 1}</div>
                <div className="product-badges">
                  {isPerishable && <span className="badge badge-perishable">Perishable</span>}
                </div>
              </div>

              {/* Inventory Level */}
              <div className="progress-group">
                <div className="progress-label">
                  <span>Inventory Level</span>
                  <strong>{inv.toFixed(0)} units</strong>
                </div>
                <div className="progress-track">
                  {/* Visual scale: max 200 */}
                  <div className={`progress-fill ${inv < 20 ? 'bg-red' : inv > 150 ? 'bg-orange' : 'bg-green'}`} style={{ width: `${Math.min((inv / 200) * 100, 100)}%` }} />
                </div>
              </div>

              {/* Expiry Warning */}
              {isPerishable && (
                <div style={{ fontSize: '0.85rem', color: days_to_expiry[idx] <= 2 ? 'var(--accent-red)' : 'var(--text-secondary)', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <AlertCircle size={14} /> Min days to expiry: <strong>{days_to_expiry[idx]}</strong>
                </div>
              )}

              {/* In Transit */}
              {incoming > 0 && (
                <div className="in-transit-container">
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', width: '100%' }}>In Transit:</div>
                  {transitArray.map((qty, d) => qty > 0 ? (
                    <span key={d} className="in-transit-pill">
                      <Truck size={12} /> {qty.toFixed(0)} in {d}d
                    </span>
                  ) : null)}
                </div>
              )}

              {/* Order Control */}
              <div className="order-controls">
                <div className="order-label">Reorder Quantity</div>
                <div className="slider-container">
                  <input 
                    type="range" 
                    className="order-slider"
                    min="0" 
                    max="100" 
                    step="5"
                    value={orderQuantities[idx] || 0}
                    onChange={(e) => handleOrderChange(idx, e.target.value)}
                  />
                  <div className="order-value">+{orderQuantities[idx] || 0}</div>
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
        >
          {score !== null ? "Episode Done" : "Execute Step & Advance Day"} <Play size={20} />
        </button>
      </div>

    </div>
  );
}

export default App;
