/**
 * Computes a rolling mean matching pandas .rolling(window, min_periods=1).mean()
 * i.e. for each index i, average of values[max(0, i-window+1)..i]
 */
export function rollingMean(values, window) {
  return values.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

/**
 * Find contiguous frame ranges where rollingValues > threshold.
 * Returns array of { x0, x1 } objects (frame numbers).
 */
export function findHighRegions(frames, rollingValues, threshold) {
  const regions = [];
  let start = null;
  for (let i = 0; i < rollingValues.length; i++) {
    if (rollingValues[i] > threshold && start === null) {
      start = frames[i];
    }
    if (rollingValues[i] <= threshold && start !== null) {
      regions.push({ x0: start, x1: frames[i - 1] });
      start = null;
    }
  }
  if (start !== null) {
    regions.push({ x0: start, x1: frames[frames.length - 1] });
  }
  return regions;
}

/**
 * Determine status emoji + label from peak aggression vs threshold.
 */
export function getStatus(peakAggr, threshold) {
  if (peakAggr < threshold * 0.8) return { emoji: '🟢', label: 'Normal' };
  if (peakAggr < threshold)       return { emoji: '🟡', label: 'Caution' };
  return                                 { emoji: '🔴', label: 'Alert' };
}
