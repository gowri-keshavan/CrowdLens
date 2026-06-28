import React, { useState, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import KpiRow from './components/KpiRow';
import AggressionChart from './components/AggressionChart';
import DensityChart from './components/DensityChart';
import { getStatus } from './utils/rolling';
import styles from './App.module.css';

export default function App() {
  // Sidebar controls — mirror Streamlit defaults exactly
  const [rollingWindow, setRollingWindow] = useState(10);
  const [aggrThreshold, setAggrThreshold] = useState(60);
  const [preset, setPreset] = useState('Balanced');

  // Analysis state
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ran, setRan] = useState(false);

  const handleRunAnalysis = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setRan(true);
    try {
      const res = await fetch('/api/analyze', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Analysis failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = async () => {
    const res = await fetch('/api/download-csv', { method: 'POST' });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crowd_stats.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Derived KPIs — recomputed when result or threshold changes
  const kpis = useMemo(() => {
    if (!result) return null;
    const peakAggr = Math.max(...result.aggression);
    return {
      totalPeople: result.total_people,
      peakAggr,
      status: getStatus(peakAggr, aggrThreshold),
    };
  }, [result, aggrThreshold]);

  return (
    <div className={styles.layout}>
      <Sidebar
        rollingWindow={rollingWindow} setRollingWindow={setRollingWindow}
        aggrThreshold={aggrThreshold} setAggrThreshold={setAggrThreshold}
        preset={preset} setPreset={setPreset}
      />

      <main className={styles.main}>
        <h1 className={styles.pageTitle}>CrowdLens – Crowd Analytics Dashboard</h1>

        {/* Run Analysis button */}
        <div className={styles.runRow}>
          <button
            className={styles.runBtn}
            onClick={handleRunAnalysis}
            disabled={loading}
          >
            {loading ? (
              <><span className={styles.spinner} /> Processing video…</>
            ) : (
              '▶  Run Analysis'
            )}
          </button>

          {result && (
            <button className={styles.csvBtn} onClick={handleDownloadCSV}>
              ⬇ Download results as CSV
            </button>
          )}
        </div>

        {/* Spinner message */}
        {loading && (
          <div className={styles.spinnerMsg}>
            Processing video… this may take a while depending on model and video length
          </div>
        )}

        {/* Error */}
        {error && (
          <div className={styles.errorBox}>⚠ {error}</div>
        )}

        {/* Success + results */}
        {result && !loading && (
          <>
            <div className={styles.successBox}>✓ Analysis finished!</div>

            <KpiRow
              totalPeople={kpis.totalPeople}
              peakAggr={kpis.peakAggr}
              status={kpis.status}
            />

            <AggressionChart
              data={result}
              rollingWindow={rollingWindow}
              aggrThreshold={aggrThreshold}
            />

            <DensityChart data={result} />
          </>
        )}

        {/* Idle state */}
        {!ran && !loading && (
          <div className={styles.idleBox}>
            Click <strong>Run Analysis</strong> to process the video and see crowd statistics and aggression graph.
          </div>
        )}
      </main>
    </div>
  );
}
