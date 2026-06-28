import React from 'react';
import styles from './KpiRow.module.css';

function KpiCard({ label, value, help }) {
  return (
    <div className={styles.card} title={help}>
      <span className={styles.label}>{label}</span>
      <span className={styles.value}>{value}</span>
      {help && <span className={styles.help}>{help}</span>}
    </div>
  );
}

export default function KpiRow({ totalPeople, peakAggr, status }) {
  return (
    <div className={styles.row}>
      <KpiCard
        label="Total Unique People"
        value={totalPeople ?? '—'}
        help="Unique track IDs seen in the video"
      />
      <KpiCard
        label="Peak Aggression (%)"
        value={peakAggr != null ? peakAggr.toFixed(1) : '—'}
      />
      <KpiCard
        label="Status"
        value={status ? `${status.emoji} ${status.label}` : '—'}
      />
    </div>
  );
}
