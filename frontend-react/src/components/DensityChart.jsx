import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';
import styles from './Chart.module.css';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipFrame}>Frame {label}</p>
      <p className={styles.tooltipLine} style={{ color: '#f5a623' }}>
        Density: {Number(payload[0].value).toExponential(3)}
      </p>
    </div>
  );
};

export default function DensityChart({ data }) {
  const chartData = data
    ? data.steps.map((frame, i) => ({ frame, density: data.density[i] }))
    : [];

  return (
    <div className={styles.chartBox}>
      <h3 className={styles.chartTitle}>Density Over Time</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" vertical={false} />
          <XAxis
            dataKey="frame"
            tick={{ fill: '#4a5270', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Frame', position: 'insideBottomRight', offset: -5, fill: '#4a5270', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: '#4a5270', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'No. of People', angle: -90, position: 'insideLeft', fill: '#4a5270', fontSize: 11, dy: 50 }}
            tickFormatter={v => v.toExponential(1)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="density"
            stroke="#f5a623"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#f5a623', stroke: '#0f1117', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
