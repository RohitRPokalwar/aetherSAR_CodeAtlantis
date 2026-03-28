import React, { useState } from 'react';
import { ChevronUp, ChevronDown, Download, Filter, ArrowUpDown, Ship, AlertTriangle, Target, FileText } from 'lucide-react';
import { Detection, DetectionSummary } from './types';

interface DetectionAnalyticsTrayProps {
  detections: Detection[];
  selectedId: string | null;
  onSelectDetection: (id: string | null) => void;
  isOpen: boolean;
  onToggle: () => void;
  summary: DetectionSummary | null;
  onExportCSV: () => void;
}

type SortKey = keyof Detection;
type SortDir = 'asc' | 'desc';

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const secs = ms / 1000;
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const mins = Math.floor(secs / 60);
  const remSecs = Math.round(secs % 60);
  return `${mins}m ${remSecs}s`;
}

export const DetectionAnalyticsTray: React.FC<DetectionAnalyticsTrayProps> = ({
  detections, selectedId, onSelectDetection, isOpen, onToggle, summary, onExportCSV,
}) => {
  const [sortKey, setSortKey] = useState<SortKey>('confidence');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [filterType, setFilterType] = useState<string>('All');
  const [minConf, setMinConf] = useState<number>(0);
  const [showFilters, setShowFilters] = useState(false);

  const sorted = [...detections]
    .filter(d => filterType === 'All' || d.type === filterType)
    .filter(d => d.confidence >= minConf / 100)
    .sort((a, b) => {
      const av = a[sortKey] as number | string;
      const bv = b[sortKey] as number | string;
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === 'asc' ? cmp : -cmp;
    });

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const types = ['All', ...Array.from(new Set(detections.map(d => d.type)))];
  const highConf = detections.filter(d => d.confidence >= 0.9).length;
  const noAIS = detections.filter(d => !d.ais).length;

  const coverageStr = summary ? `${summary.coverageKm2} km²` : '—';
  const procTimeStr = summary ? formatTime(summary.processingTimeMs) : '—';

  const ColHeader: React.FC<{ label: string; sk: SortKey; className?: string }> = ({ label, sk, className }) => (
    <th className={`text-left pb-2 cursor-pointer select-none ${className ?? ''}`}
      onClick={() => handleSort(sk)}>
      <div className="flex items-center gap-1 text-xs uppercase tracking-wider"
        style={{ color: sortKey === sk ? '#00c8ff' : '#334155' }}>
        {label}
        <ArrowUpDown className="h-2.5 w-2.5 flex-none" />
      </div>
    </th>
  );

  return (
    <div className="flex-none flex flex-col transition-all duration-300"
      style={{ height: isOpen ? '280px' : '36px', borderTop: '1px solid #1a2d45', background: '#08121e' }}>

      {/* Tray header */}
      <div className="flex-none flex items-center justify-between px-4 h-9 cursor-pointer border-b"
        style={{ borderColor: '#1a2d45' }}
        onClick={onToggle}>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1" style={{ color: '#475569' }}>
            {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
          </button>
          <span className="text-xs font-bold tracking-widest uppercase" style={{ color: '#94a3b8' }}>
            Detection Analytics
          </span>
          <div className="px-2 py-0.5 rounded text-xs"
            style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.15)', color: '#00c8ff' }}>
            {detections.length} targets
          </div>
        </div>

        {isOpen && (
          <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
            <button onClick={() => setShowFilters(v => !v)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors"
              style={{
                background: showFilters ? 'rgba(0,200,255,0.1)' : '#0a1525',
                border: `1px solid ${showFilters ? 'rgba(0,200,255,0.3)' : '#1a2d45'}`,
                color: showFilters ? '#00c8ff' : '#475569',
              }}>
              <Filter className="h-3 w-3" /> Filter
            </button>
            <button onClick={onExportCSV}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors"
              style={{ background: '#0a1525', border: '1px solid #1a2d45', color: '#475569' }}
              onMouseEnter={e => { e.currentTarget.style.color = '#22c55e'; e.currentTarget.style.borderColor = 'rgba(34,197,94,0.3)'; }}
              onMouseLeave={e => { e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = '#1a2d45'; }}>
              <Download className="h-3 w-3" /> Export CSV
            </button>
          </div>
        )}
      </div>

      {isOpen && (
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Summary stats — clean text only, no colored boxes */}
          <div className="flex-none flex items-center gap-6 px-4 py-2 border-b" style={{ borderColor: '#0f1d30' }}>
            <div className="flex items-center gap-2">
              <Target className="h-3 w-3" style={{ color: '#00c8ff' }} />
              <span className="text-sm font-bold" style={{ color: '#00c8ff' }}>{detections.length}</span>
              <span className="text-xs" style={{ color: '#475569' }}>Total</span>
            </div>
            <div className="w-px h-4" style={{ background: '#1a2d45' }} />
            <div className="flex items-center gap-2">
              <Ship className="h-3 w-3" style={{ color: '#22c55e' }} />
              <span className="text-sm font-bold" style={{ color: '#22c55e' }}>{highConf}</span>
              <span className="text-xs" style={{ color: '#475569' }}>High Conf</span>
            </div>
            <div className="w-px h-4" style={{ background: '#1a2d45' }} />
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-3 w-3" style={{ color: '#ef4444' }} />
              <span className="text-sm font-bold" style={{ color: '#ef4444' }}>{noAIS}</span>
              <span className="text-xs" style={{ color: '#475569' }}>Dark Vessels</span>
            </div>
            <div className="w-px h-4" style={{ background: '#1a2d45' }} />
            <span className="text-xs" style={{ color: '#475569' }}>Coverage: <span className="font-mono" style={{ color: '#64748b' }}>{coverageStr}</span></span>
            <div className="w-px h-4" style={{ background: '#1a2d45' }} />
            <span className="text-xs" style={{ color: '#475569' }}>Time: <span className="font-mono" style={{ color: '#64748b' }}>{procTimeStr}</span></span>
          </div>

          {/* Filter row */}
          {showFilters && (
            <div className="flex-none flex items-center gap-4 px-4 py-2 border-b" style={{ borderColor: '#0f1d30', background: '#060e1a' }}>
              <div className="flex items-center gap-2">
                <span className="text-xs" style={{ color: '#334155' }}>Type:</span>
                <div className="flex gap-1">
                  {types.map(t => (
                    <button key={t} onClick={() => setFilterType(t)}
                      className="px-2 py-0.5 rounded text-xs transition-colors"
                      style={{
                        background: filterType === t ? 'rgba(0,200,255,0.15)' : '#0a1525',
                        border: `1px solid ${filterType === t ? 'rgba(0,200,255,0.3)' : '#1a2d45'}`,
                        color: filterType === t ? '#00c8ff' : '#475569',
                      }}>{t}</button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs" style={{ color: '#334155' }}>Min Conf:</span>
                <input type="range" min={0} max={90} step={5} value={minConf}
                  onChange={e => setMinConf(Number(e.target.value))}
                  className="w-24 h-1 accent-cyan-400" />
                <span className="text-xs font-mono" style={{ color: '#00c8ff' }}>{minConf}%</span>
              </div>
            </div>
          )}

          {/* Table — clean, no colored badges */}
          <div className="flex-1 overflow-auto">
            <table className="w-full">
              <thead className="sticky top-0" style={{ background: '#060e1a' }}>
                <tr className="px-4">
                  <th className="w-8 pb-2" />
                  <ColHeader label="ID" sk="id" className="pl-4 w-24" />
                  <ColHeader label="Type" sk="type" className="w-24" />
                  <ColHeader label="Conf" sk="confidence" className="w-20" />
                  <ColHeader label="Length" sk="lengthM" className="w-20" />
                  <ColHeader label="Heading" sk="headingDeg" className="w-20" />
                  <ColHeader label="RCS" sk="rcs" className="w-20" />
                  <th className="pb-2 pr-4 text-left text-xs uppercase tracking-wider" style={{ color: '#334155' }}>AIS</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(det => {
                  const isSelected = det.id === selectedId;
                  return (
                    <tr
                      key={det.id}
                      onClick={() => onSelectDetection(isSelected ? null : det.id)}
                      className="cursor-pointer transition-colors"
                      style={{
                        background: isSelected ? 'rgba(0,200,255,0.08)' : 'transparent',
                        borderBottom: '1px solid #0a1525',
                      }}
                      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(0,200,255,0.04)'; }}
                      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent'; }}>
                      <td className="py-2 pl-4">
                        <div className="w-2 h-2 rounded-full" style={{ background: isSelected ? '#00c8ff' : 'transparent', border: `1px solid ${isSelected ? '#00c8ff' : '#1a2d45'}` }} />
                      </td>
                      <td className="py-2">
                        <span className="text-xs font-mono" style={{ color: isSelected ? '#00c8ff' : '#64748b' }}>{det.id}</span>
                      </td>
                      <td className="py-2">
                        <span className="text-xs" style={{ color: '#94a3b8' }}>{det.type}</span>
                      </td>
                      <td className="py-2">
                        <span className="text-xs font-mono" style={{ color: '#94a3b8' }}>
                          {Math.round(det.confidence * 100)}%
                        </span>
                      </td>
                      <td className="py-2">
                        <span className="text-xs font-mono" style={{ color: '#64748b' }}>{det.lengthM}m</span>
                      </td>
                      <td className="py-2">
                        <span className="text-xs font-mono" style={{ color: '#475569' }}>{det.headingDeg}°</span>
                      </td>
                      <td className="py-2">
                        <span className="text-xs font-mono" style={{ color: '#475569' }}>{det.rcs.toFixed(1)} dBsm</span>
                      </td>
                      <td className="py-2 pr-4">
                        {det.ais ? (
                          <span className="text-xs font-mono" style={{ color: '#22c55e' }}>{det.ais}</span>
                        ) : (
                          <span className="text-xs" style={{ color: '#ef4444' }}>DARK</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {sorted.length === 0 && (
              <div className="text-center py-6 text-xs" style={{ color: '#334155' }}>No detections match filter criteria</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
