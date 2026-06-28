import React, { useState } from 'react';
import styles from './Sidebar.module.css';

const PRESETS = ['Balanced', 'Demonstration (High)', 'Strict (Sensitive)'];

export default function Sidebar({ rollingWindow, setRollingWindow, aggrThreshold, setAggrThreshold, preset, setPreset }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <span className={styles.logoMark}>⬡</span>
        <span className={styles.logoName}>CrowdLens</span>
      </div>

      <h2 className={styles.sectionTitle}>Settings &amp; Controls</h2>

      <label className={styles.label}>
        Rolling window for smoothing (frames)
        <span className={styles.value}>{rollingWindow}</span>
      </label>
      <input
        type="range"
        min={1} max={200}
        value={rollingWindow}
        onChange={e => setRollingWindow(Number(e.target.value))}
        className={styles.slider}
      />
      <div className={styles.sliderBounds}><span>1</span><span>200</span></div>

      <label className={styles.label} style={{ marginTop: 20 }}>
        Aggression threshold (%)
        <span className={styles.value}>{aggrThreshold}%</span>
      </label>
      <input
        type="range"
        min={0} max={100}
        value={aggrThreshold}
        onChange={e => setAggrThreshold(Number(e.target.value))}
        className={styles.slider}
      />
      <div className={styles.sliderBounds}><span>0</span><span>100</span></div>

      <label className={styles.label} style={{ marginTop: 20 }}>
        Sensitivity preset
      </label>
      <select
        className={styles.select}
        value={preset}
        onChange={e => setPreset(e.target.value)}
      >
        {PRESETS.map(p => <option key={p}>{p}</option>)}
      </select>

      <hr className={styles.divider} />

      <p className={styles.hint}>
        Use the controls to adjust smoothing and the threshold used for highlighting
        high-aggression regions. This does not change the detection backend, only
        the visualization and interpretation shown here.
      </p>

      <div className={styles.expander}>
        <button className={styles.expandBtn} onClick={() => setExpanded(e => !e)}>
          {expanded ? '▾' : '▸'} How to interpret the results
        </button>
        {expanded && (
          <ul className={styles.expandContent}>
            <li>Aggression (%) is a scene-level score derived from per-person speeds (pixels/frame), smoothed for stability.</li>
            <li>Density is people per pixel (low-level proxy).</li>
            <li>Use the rolling window slider to reduce noise; higher window = smoother curve.</li>
            <li>Shaded regions indicate where aggression exceeds the chosen threshold.</li>
            <li>Watch out for camera motion or background activity — these can produce false positives.</li>
          </ul>
        )}
      </div>
    </aside>
  );
}
