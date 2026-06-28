import React, { useMemo } from 'react';
import {
  ComposedChart, Line, ReferenceLine, ReferenceArea,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { rollingMean, findHighRegions } from '../utils/rolling';
import styles from './Chart.module.css';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipFrame}>Frame {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }} className={styles.tooltipLine}>
          {p.name}: {Number(p.value).toFixed(2)}%
        </p>
      ))}
    </div>
  );
};

export default function AggressionChart({ data, rollingWindow, aggrThreshold }) {
  const chartData = useMemo(() => {
    if (!data) return [];
    const rolling = rollingMean(data.aggression, rollingWindow);
    return data.steps.map((frame, i) => ({
      frame,
      raw: data.aggression[i],
      rolling: rolling[i],
    }));
  }, [data, rollingWindow]);

  const highRegions = useMemo(() => {
    if (!data) return [];
    const rolling = rollingMean(data.aggression, rollingWindow);
    return findHighRegions(data.steps, rolling, aggrThreshold);
  }, [data, rollingWindow, aggrThreshold]);

  return (
    <div className={styles.chartBox}>
      <h3 className={styles.chartTitle}>Aggression Score Over Time</h3>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" vertical={false} />
          <XAxis
            dataKey="frame"
            tick={{ fill: '#4a5270', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Frame', position: 'insideBottomRight', offset: -5, fill: '#4a5270', fontSize: 11 }}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: '#4a5270', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Aggression (%)', angle: -90, position: 'insideLeft', fill: '#4a5270', fontSize: 11, dy: 60 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: '#6b7394', paddingTop: 8 }}
          />

          {/* Red shaded regions where rolling > threshold */}
          {highRegions.map((r, i) => (
            <ReferenceArea
              key={i}
              x1={r.x0}
              x2={r.x1}
              fill="#e8394a"
              fillOpacity={0.08}
              stroke="none"
            />
          ))}

          {/* Horizontal threshold line */}
          <ReferenceLine
            y={aggrThreshold}
            stroke="#f39c12"
            strokeDasharray="6 3"
            label={{
              value: `Threshold (${aggrThreshold}%)`,
              position: 'insideTopLeft',
              fill: '#f39c12',
              fontSize: 11,
              dy: -6,
            }}
          />

          {/* Raw aggression - light gray */}
          <Line
            type="monotone"
            dataKey="raw"
            name="Raw"
            stroke="#4a5270"
            strokeWidth={1}
            dot={false}
            activeDot={false}
          />

          {/* Rolling mean - firebrick, bold */}
          <Line
            type="monotone"
            dataKey="rolling"
            name={`Rolling (w=${rollingWindow})`}
            stroke="#c0392b"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4, fill: '#c0392b', stroke: '#0f1117', strokeWidth: 2 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
