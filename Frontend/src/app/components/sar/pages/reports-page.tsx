import React, { useMemo, useState } from 'react';
import { FileText, Download, Clock, Grid3X3, Target, Ship, MapPin } from 'lucide-react';
import { Detection, DetectionSummary } from '../types';
import Particles from '../Particles';
import { TargetGalleryModal } from '../target-gallery-modal';

interface ReportsPageProps {
  detections: Detection[];
  summary: DetectionSummary | null;
  fileId: string | null;
  darkMode: boolean;
}

/* ── Helpers ── */
function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

/* ── Donut Chart SVG ── */
const DonutChart: React.FC<{ data: { label: string; value: number; color: string }[]; darkMode: boolean }> = ({ data, darkMode }) => {
  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return null;
  const r = 50;
  const cx = 65;
  const cy = 65;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  return (
    <div className="flex items-center gap-6">
      <svg width="130" height="130" viewBox="0 0 130 130">
        {data.map((d, i) => {
          const pct = d.value / total;
          const dashLen = pct * circumference;
          const dashOffset = -offset;
          offset += dashLen;
          return (
            <circle key={i} cx={cx} cy={cy} r={r}
              fill="none" stroke={d.color} strokeWidth="16"
              strokeDasharray={`${dashLen} ${circumference - dashLen}`}
              strokeDashoffset={dashOffset}
              transform={`rotate(-90 ${cx} ${cy})`}
            />
          );
        })}
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize="18" fontWeight="bold"
          fill={darkMode ? '#e2e8f0' : '#1e293b'}>{total}</text>
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize="9"
          fill={darkMode ? '#475569' : '#94a3b8'}>Total</text>
      </svg>
      <div className="space-y-2">
        {data.map((d, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: d.color }} />
            <span className="text-xs" style={{ color: darkMode ? '#94a3b8' : '#475569' }}>{d.label}</span>
            <span className="text-xs font-mono font-bold" style={{ color: d.color }}>{d.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ── Auto-Generated Summary Text ── */
function generateSummaryText(dets: Detection[], summary: DetectionSummary | null): string {
  if (dets.length === 0) return 'No detection results available. Upload SAR imagery and run the detection pipeline to generate a report.';

  const avg = Math.round((dets.reduce((s, d) => s + d.confidence, 0) / dets.length) * 100);
  const darkCount = dets.filter(d => !d.ais).length;
  const highConf = dets.filter(d => d.confidence >= 0.9).length;

  // Quadrant analysis
  const geoDetections = dets.filter(d => d.lat != null && d.lon != null);
  const midLat = geoDetections.length > 0 ? geoDetections.reduce((s, d) => s + (d.lat ?? 0), 0) / geoDetections.length : 0;
  const midLon = geoDetections.length > 0 ? geoDetections.reduce((s, d) => s + (d.lon ?? 0), 0) / geoDetections.length : 0;
  const q: Record<string, number> = { NE: 0, NW: 0, SE: 0, SW: 0 };
  geoDetections.forEach(d => {
    const ns = (d.lat ?? 0) >= midLat ? 'N' : 'S';
    const ew = (d.lon ?? 0) >= midLon ? 'E' : 'W';
    q[ns + ew]++;
  });
  const densest = Object.entries(q).sort((a, b) => b[1] - a[1])[0];

  const parts = [
    `SAR detection analysis identified ${dets.length} maritime target${dets.length !== 1 ? 's' : ''} with an average confidence of ${avg}%.`,
    highConf > 0 ? `${highConf} target${highConf !== 1 ? 's' : ''} exceeded the 90% confidence threshold.` : '',
    darkCount > 0 ? `⚠ ${darkCount} vessel${darkCount !== 1 ? 's' : ''} operating without AIS transponders (dark vessels) — flagged for review.` : 'All detected vessels are broadcasting AIS.',
    densest ? `Highest ship concentration observed in the ${densest[0]} quadrant (${densest[1]} vessels).` : '',
    summary ? `Processing covered ${summary.coverageKm2} km² in ${formatTime(summary.processingTimeMs)} across ${summary.tilesProcessed} tiles.` : '',
  ];

  return parts.filter(Boolean).join(' ');
}

/* ── Confidence Bar Chart (reuse pattern) ── */
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
  const chartH = 100;

  return (
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
  );
};

/* ── Main Reports Page ── */
export const ReportsPage: React.FC<ReportsPageProps> = ({ detections, summary, fileId, darkMode }) => {
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const summaryText = useMemo(() => generateSummaryText(detections, summary), [detections, summary]);

  // Ship type data for donut chart
  const typeData = useMemo(() => {
    const counts: Record<string, number> = {};
    detections.forEach(d => { counts[d.type] = (counts[d.type] || 0) + 1; });
    const colors = ['#00c8ff', '#22c55e', '#f59e0b', '#a78bfa', '#ef4444', '#f472b6'];
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([label, value], i) => ({ label, value, color: colors[i % colors.length] }));
  }, [detections]);

  const cardBg = darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const labelColor = darkMode ? '#475569' : '#94a3b8';
  const textColor = darkMode ? '#e2e8f0' : '#1e293b';

  const handleDownloadPDF = async () => {
    if (!fileId) return;
    try {
      const response = await fetch(`/api/report/${fileId}`);
      if (!response.ok) throw new Error('Failed to generate report');
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SAR_Report_${fileId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Report download failed:', err);
    }
  };

  return (
    <div className="flex-1 overflow-hidden relative" style={{ background: darkMode ? '#040c18' : '#f8fafc' }}>

      {/* Particle Background */}
      <Particles
        particleColors={['#6366f1', '#a78bfa', '#00c8ff']}
        particleCount={100}
        particleSpread={10}
        speed={0.04}
        particleBaseSize={150}
        alphaParticles
        sizeRandomness={1}
        cameraDistance={26}
        disableRotation={false}
      />

      <div className="relative z-10 overflow-y-auto h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold tracking-wide" style={{ color: textColor }}>Mission Report</h2>
          <p className="text-xs mt-1" style={{ color: labelColor }}>Auto-generated detection summary and analytics</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setIsGalleryOpen(true)}
            disabled={!fileId || detections.length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all disabled:opacity-40"
            style={{
              background: fileId && detections.length > 0 ? 'linear-gradient(135deg, #00c8ff, #0066cc)' : (darkMode ? '#0a1525' : '#e2e8f0'),
              color: '#fff',
              boxShadow: fileId && detections.length > 0 ? '0 0 20px rgba(0,200,255,0.3)' : 'none',
            }}
          >
            <Target className="h-4 w-4" />
            View Target Gallery
          </button>
          <button
            onClick={handleDownloadPDF}
            disabled={!fileId || detections.length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all disabled:opacity-40"
            style={{
              background: fileId && detections.length > 0 ? 'linear-gradient(135deg, #6366f1, #a78bfa)' : (darkMode ? '#0a1525' : '#e2e8f0'),
              color: '#fff',
              boxShadow: fileId && detections.length > 0 ? '0 0 20px rgba(99,102,241,0.3)' : 'none',
            }}
          >
            <Download className="h-4 w-4" />
            Download PDF Report
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="rounded-lg p-5" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
        <div className="flex items-center gap-2 mb-3">
          <FileText className="h-4 w-4" style={{ color: '#00c8ff' }} />
          <span className="text-xs font-bold uppercase tracking-widest" style={{ color: labelColor }}>Intelligence Summary</span>
        </div>
        <p className="text-sm leading-relaxed" style={{ color: darkMode ? '#94a3b8' : '#475569' }}>{summaryText}</p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { icon: <Target className="h-4 w-4" />, label: 'Detections', value: String(detections.length), color: '#00c8ff' },
          { icon: <Grid3X3 className="h-4 w-4" />, label: 'Tiles', value: summary ? String(summary.tilesProcessed) : '—', color: '#a78bfa' },
          { icon: <Clock className="h-4 w-4" />, label: 'Time', value: summary ? formatTime(summary.processingTimeMs) : '—', color: '#f59e0b' },
          { icon: <MapPin className="h-4 w-4" />, label: 'Coverage', value: summary ? `${summary.coverageKm2} km²` : '—', color: '#22c55e' },
        ].map(({ icon, label, value, color }) => (
          <div key={label} className="p-4 rounded-lg" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
            <div className="flex items-center gap-2 mb-2">
              <span style={{ color }}>{icon}</span>
              <span className="text-xs uppercase tracking-widest" style={{ color: labelColor }}>{label}</span>
            </div>
            <span className="text-xl font-bold font-mono" style={{ color }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Confidence Distribution */}
        <div className="rounded-lg p-4" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
          <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: labelColor }}>
            Confidence Distribution
          </div>
          {detections.length === 0 ? (
            <div className="flex items-center justify-center h-28 text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No data</div>
          ) : (
            <ConfidenceChart detections={detections} darkMode={darkMode} />
          )}
        </div>

        {/* Ship Type Distribution */}
        <div className="rounded-lg p-4" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
          <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: labelColor }}>
            Ship Type Distribution
          </div>
          {detections.length === 0 ? (
            <div className="flex items-center justify-center h-28 text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No data</div>
          ) : (
            <DonutChart data={typeData} darkMode={darkMode} />
          )}
        </div>
      </div>

      {/* Detection Table Preview */}
      <div className="rounded-lg overflow-hidden" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
        <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: borderC }}>
          <div className="flex items-center gap-2">
            <Ship className="h-4 w-4" style={{ color: '#00c8ff' }} />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: labelColor }}>Detection Log</span>
          </div>
          <span className="text-xs" style={{ color: labelColor }}>{detections.length} entries</span>
        </div>
        <div className="max-h-[250px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0" style={{ background: darkMode ? '#060e1a' : '#f8fafc' }}>
              <tr>
                <th className="text-left px-4 py-2 text-xs uppercase tracking-wider" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>ID</th>
                <th className="text-left px-4 py-2 text-xs uppercase tracking-wider" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>Type</th>
                <th className="text-left px-4 py-2 text-xs uppercase tracking-wider" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>Conf</th>
                <th className="text-left px-4 py-2 text-xs uppercase tracking-wider" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>Position</th>
                <th className="text-left px-4 py-2 text-xs uppercase tracking-wider" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>AIS</th>
              </tr>
            </thead>
            <tbody>
              {detections.slice(0, 20).map(det => (
                <tr key={det.id} className="border-t" style={{ borderColor: darkMode ? '#0a1525' : '#f1f5f9' }}>
                  <td className="px-4 py-2 text-xs font-mono" style={{ color: darkMode ? '#64748b' : '#94a3b8' }}>{det.id}</td>
                  <td className="px-4 py-2 text-xs" style={{ color: darkMode ? '#94a3b8' : '#475569' }}>{det.type}</td>
                  <td className="px-4 py-2 text-xs font-mono" style={{ color: det.confidence >= 0.9 ? '#22c55e' : '#f59e0b' }}>{Math.round(det.confidence * 100)}%</td>
                  <td className="px-4 py-2 text-xs font-mono" style={{ color: darkMode ? '#475569' : '#94a3b8' }}>{det.lat != null ? `${det.lat.toFixed(4)}°N` : 'N/A'}, {det.lon != null ? `${det.lon.toFixed(4)}°E` : 'N/A'}</td>
                  <td className="px-4 py-2 text-xs" style={{ color: det.ais ? '#22c55e' : '#ef4444' }}>{det.ais || 'DARK'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {detections.length === 0 && (
            <div className="text-center py-8 text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No detections to report</div>
          )}
          {detections.length > 20 && (
            <div className="text-center py-3 text-xs" style={{ color: labelColor }}>
              Showing 20 of {detections.length} — download PDF for full report
            </div>
          )}
        </div>
      </div>
      </div>
      <TargetGalleryModal
        isOpen={isGalleryOpen}
        onClose={() => setIsGalleryOpen(false)}
        fileId={fileId}
        detections={detections}
        darkMode={darkMode}
      />
    </div>
  );
};
