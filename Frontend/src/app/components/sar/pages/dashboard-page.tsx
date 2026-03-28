import React from 'react';
import { Target, TrendingUp, Clock, Grid3X3, Ship } from 'lucide-react';
import { Detection, DetectionSummary } from '../types';
import Particles from '../Particles';

interface DashboardPageProps {
  detections: Detection[];
  summary: DetectionSummary | null;
  darkMode: boolean;
}

/* ── Helpers ── */
function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function avgConfidence(dets: Detection[]): number {
  if (dets.length === 0) return 0;
  return dets.reduce((s, d) => s + d.confidence, 0) / dets.length;
}

/* ── Summary Card ── */
const StatCard: React.FC<{
  icon: React.ReactNode; label: string; value: string; accent: string; darkMode: boolean;
}> = ({ icon, label, value, accent, darkMode }) => (
  <div
    className="flex flex-col gap-2 p-4 rounded-lg"
    style={{
      background: darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)',
      backdropFilter: 'blur(8px)',
      border: `1px solid ${darkMode ? '#1a2d45' : '#e2e8f0'}`,
    }}
  >
    <div className="flex items-center gap-2">
      <div className="w-8 h-8 rounded-lg flex items-center justify-center"
        style={{ background: `${accent}15`, color: accent }}>{icon}</div>
      <span className="text-xs uppercase tracking-widest" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>{label}</span>
    </div>
    <span className="text-2xl font-bold font-mono" style={{ color: accent }}>{value}</span>
  </div>
);

/* ── Confidence Distribution Bar Chart (SVG) ── */
const ConfidenceChart: React.FC<{ detections: Detection[]; darkMode: boolean }> = ({ detections, darkMode }) => {
  const buckets = Array.from({ length: 10 }, (_, i) => {
    const lo = i * 0.1;
    const hi = lo + 0.1;
    return detections.filter(d => d.confidence >= lo && d.confidence < (i === 9 ? 1.01 : hi)).length;
  });
  const max = Math.max(...buckets, 1);
  const barW = 28;
  const gap = 8;
  const chartW = buckets.length * (barW + gap);
  const chartH = 120;

  return (
    <div className="p-4 rounded-lg" style={{ background: darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)',
      backdropFilter: 'blur(8px)', border: `1px solid ${darkMode ? '#1a2d45' : '#e2e8f0'}` }}>
      <div className="text-xs uppercase tracking-widest mb-3" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>
        Confidence Distribution
      </div>
      {detections.length === 0 ? (
        <div className="flex items-center justify-center h-28 text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No detection data</div>
      ) : (
        <svg width="100%" viewBox={`0 0 ${chartW} ${chartH + 20}`} preserveAspectRatio="xMidYMid meet">
          {buckets.map((count, i) => {
            const h = (count / max) * chartH;
            const x = i * (barW + gap);
            const y = chartH - h;
            return (
              <g key={i}>
                <rect x={x} y={y} width={barW} height={h} rx={4}
                  fill={`rgba(0,200,255,${0.3 + (i / 10) * 0.7})`} />
                {count > 0 && (
                  <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize="9"
                    fill={darkMode ? '#94a3b8' : '#64748b'}>{count}</text>
                )}
                <text x={x + barW / 2} y={chartH + 14} textAnchor="middle" fontSize="8"
                  fill={darkMode ? '#334155' : '#94a3b8'}>{i * 10}%</text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
};

/* ── Detections Over Time (simple SVG line chart placeholder) ── */
const TimelineChart: React.FC<{ detections: Detection[]; darkMode: boolean }> = ({ detections, darkMode }) => {
  // We group detections by confidence bands as a proxy for "time" since we don't have real timestamps
  // Show a cumulative detection curve
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const points = sorted.map((_, i) => ({ x: i, y: i + 1 }));
  const chartW = 360;
  const chartH = 100;

  return (
    <div className="p-4 rounded-lg" style={{ background: darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)',
      backdropFilter: 'blur(8px)', border: `1px solid ${darkMode ? '#1a2d45' : '#e2e8f0'}` }}>
      <div className="text-xs uppercase tracking-widest mb-3" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>
        Cumulative Detections
      </div>
      {detections.length === 0 ? (
        <div className="flex items-center justify-center h-28 text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No detection data</div>
      ) : (
        <svg width="100%" viewBox={`0 0 ${chartW} ${chartH + 10}`} preserveAspectRatio="xMidYMid meet">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(f => (
            <line key={f} x1={0} y1={chartH * (1 - f)} x2={chartW} y2={chartH * (1 - f)}
              stroke={darkMode ? '#1a2d45' : '#e2e8f0'} strokeWidth={0.5} />
          ))}
          {/* Area fill */}
          <path
            d={`M0,${chartH} ${points.map(p => `L${(p.x / (points.length - 1 || 1)) * chartW},${chartH - (p.y / points.length) * chartH}`).join(' ')} L${chartW},${chartH} Z`}
            fill="rgba(0,200,255,0.08)"
          />
          {/* Line */}
          <path
            d={points.map((p, i) => `${i === 0 ? 'M' : 'L'}${(p.x / (points.length - 1 || 1)) * chartW},${chartH - (p.y / points.length) * chartH}`).join(' ')}
            fill="none" stroke="#00c8ff" strokeWidth={2}
          />
          {/* End dot */}
          {points.length > 0 && (
            <circle cx={chartW} cy={chartH - chartH} r={3} fill="#00c8ff" />
          )}
          <text x={chartW - 4} y={8} textAnchor="end" fontSize="9" fill="#00c8ff">
            {detections.length}
          </text>
        </svg>
      )}
    </div>
  );
};

/* ── Ship Type Breakdown ── */
const TypeBreakdown: React.FC<{ detections: Detection[]; darkMode: boolean }> = ({ detections, darkMode }) => {
  const counts: Record<string, number> = {};
  detections.forEach(d => { counts[d.type] = (counts[d.type] || 0) + 1; });
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const colors = ['#00c8ff', '#22c55e', '#f59e0b', '#a78bfa', '#ef4444', '#f472b6'];

  return (
    <div className="p-4 rounded-lg" style={{ background: darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)',
      backdropFilter: 'blur(8px)', border: `1px solid ${darkMode ? '#1a2d45' : '#e2e8f0'}` }}>
      <div className="text-xs uppercase tracking-widest mb-3" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>
        Ship Type Breakdown
      </div>
      {entries.length === 0 ? (
        <div className="text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No data</div>
      ) : (
        <div className="space-y-2">
          {entries.map(([type, count], i) => {
            const pct = (count / detections.length) * 100;
            const c = colors[i % colors.length];
            return (
              <div key={type}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs" style={{ color: darkMode ? '#94a3b8' : '#475569' }}>{type}</span>
                  <span className="text-xs font-mono" style={{ color: c }}>{count} ({Math.round(pct)}%)</span>
                </div>
                <div className="h-1.5 rounded-full" style={{ background: darkMode ? '#0f1d30' : '#e2e8f0' }}>
                  <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: c }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

/* ── Main Dashboard Page ── */
export const DashboardPage: React.FC<DashboardPageProps> = ({ detections, summary, darkMode }) => {
  const avg = avgConfidence(detections);

  return (
    <div className="flex-1 overflow-hidden relative"
      style={{ background: darkMode ? '#040c18' : '#f8fafc' }}>

      {/* Particle Background */}
      <Particles
        particleColors={['#00c8ff', '#0066cc', '#1e3a5f']}
        particleCount={120}
        particleSpread={10}
        speed={0.05}
        particleBaseSize={150}
        alphaParticles
        sizeRandomness={1}
        cameraDistance={25}
        disableRotation={false}
      />

      <div className="relative z-10 overflow-y-auto h-full p-6 space-y-6">

      {/* Header */}
      <div>
        <h2 className="text-lg font-bold tracking-wide" style={{ color: darkMode ? '#e2e8f0' : '#1e293b' }}>
          Mission Overview
        </h2>
        <p className="text-xs mt-1" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>
          Real-time detection intelligence and analytics
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          icon={<Target className="h-4 w-4" />}
          label="Total Detections"
          value={String(detections.length)}
          accent="#00c8ff"
          darkMode={darkMode}
        />
        <StatCard
          icon={<TrendingUp className="h-4 w-4" />}
          label="Avg Confidence"
          value={detections.length > 0 ? `${Math.round(avg * 100)}%` : '—'}
          accent="#22c55e"
          darkMode={darkMode}
        />
        <StatCard
          icon={<Clock className="h-4 w-4" />}
          label="Processing Time"
          value={summary ? formatTime(summary.processingTimeMs) : '—'}
          accent="#f59e0b"
          darkMode={darkMode}
        />
        <StatCard
          icon={<Grid3X3 className="h-4 w-4" />}
          label="Tiles Processed"
          value={summary ? String(summary.tilesProcessed) : '—'}
          accent="#a78bfa"
          darkMode={darkMode}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        <ConfidenceChart detections={detections} darkMode={darkMode} />
        <TimelineChart detections={detections} darkMode={darkMode} />
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-2 gap-4">
        <TypeBreakdown detections={detections} darkMode={darkMode} />

        {/* Quick Stats */}
        <div className="p-4 rounded-lg" style={{ background: darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)',
      backdropFilter: 'blur(8px)', border: `1px solid ${darkMode ? '#1a2d45' : '#e2e8f0'}` }}>
          <div className="text-xs uppercase tracking-widest mb-3" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>
            Quick Stats
          </div>
          <div className="space-y-3">
            {[
              { label: 'High Confidence (≥90%)', value: detections.filter(d => d.confidence >= 0.9).length, color: '#22c55e' },
              { label: 'Medium Confidence (50–89%)', value: detections.filter(d => d.confidence >= 0.5 && d.confidence < 0.9).length, color: '#f59e0b' },
              { label: 'Low Confidence (<50%)', value: detections.filter(d => d.confidence < 0.5).length, color: '#ef4444' },
              { label: 'Dark Vessels (No AIS)', value: detections.filter(d => !d.ais).length, color: '#ef4444' },
              { label: 'Coverage', value: summary ? `${summary.coverageKm2} km²` : '—', color: '#00c8ff' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-center justify-between">
                <span className="text-xs" style={{ color: darkMode ? '#64748b' : '#94a3b8' }}>{label}</span>
                <span className="text-sm font-mono font-bold" style={{ color }}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      </div>
    </div>
  );
};
