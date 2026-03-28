import React, { useState } from 'react';
import { Detection } from '../types';
import { MapPin, Target, Ship, Download, ArrowUpDown } from 'lucide-react';

interface LocationPageProps {
  detections: Detection[];
  fileId: string | null;
  darkMode: boolean;
}

type SortKey = 'id' | 'lat' | 'confidence' | 'type';
type SortDir = 'asc' | 'desc';

export const LocationPage: React.FC<LocationPageProps> = ({ detections, fileId, darkMode }) => {
  const [sortKey, setSortKey] = useState<SortKey>('confidence');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const textC = darkMode ? '#e2e8f0' : '#1e293b';
  const labelC = darkMode ? '#94a3b8' : '#64748b';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const cardBg = darkMode ? '#0a1525' : '#ffffff';
  const headerBg = darkMode ? '#060e1a' : '#f1f5f9';

  const sorted = [...detections].sort((a, b) => {
    let av: any = a[sortKey];
    let bv: any = b[sortKey];
    if (sortKey === 'lat') {
      av = a.lat !== null ? a.lat : (a.pixelCenter ? a.pixelCenter[0] : 0);
      bv = b.lat !== null ? b.lat : (b.pixelCenter ? b.pixelCenter[0] : 0);
    }
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortDir === 'asc' ? cmp : -cmp;
  });

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const handleExportCSV = () => {
    if (!fileId) return;
    window.open(`/api/export/${fileId}`, '_blank');
  };

  const ColHeader: React.FC<{ label: string; sk: SortKey; className?: string }> = ({ label, sk, className }) => (
    <th className={`text-left pb-3 pt-3 px-4 cursor-pointer select-none border-b ${className ?? ''}`}
      style={{ borderColor: borderC }}
      onClick={() => handleSort(sk)}>
      <div className="flex items-center gap-1 text-xs uppercase tracking-wider"
        style={{ color: sortKey === sk ? '#00c8ff' : labelC }}>
        {label}
        <ArrowUpDown className="h-2.5 w-2.5 flex-none" />
      </div>
    </th>
  );

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 py-5 flex-none flex items-center justify-between border-b" style={{ borderColor: borderC, background: cardBg }}>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center shadow-lg"
            style={{ background: 'linear-gradient(135deg, rgba(0,200,255,0.1), rgba(0,100,200,0.1))', border: '1px solid rgba(0,200,255,0.2)' }}>
            <MapPin className="h-6 w-6" style={{ color: '#00c8ff' }} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight flex items-center gap-2" style={{ color: textC }}>
              Ship Location Index
            </h1>
            <p className="text-sm mt-0.5" style={{ color: labelC }}>
              Geospatial and pixel-coordinate tracking matrix for all acquired targets.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border" style={{ background: headerBg, borderColor: borderC }}>
            <Target className="h-4 w-4" style={{ color: '#00c8ff' }} />
            <span className="text-sm font-bold font-mono" style={{ color: textC }}>{detections.length}</span>
            <span className="text-xs" style={{ color: labelC }}>Targets</span>
          </div>
          <button onClick={handleExportCSV}
            disabled={!fileId || detections.length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: 'linear-gradient(135deg, #00c8ff, #0066cc)',
              color: '#fff',
              boxShadow: fileId && detections.length > 0 ? '0 0 15px rgba(0,200,255,0.3)' : 'none',
            }}>
            <Download className="h-4 w-4" /> Export CSV
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6" style={{ background: darkMode ? '#040c18' : '#f8fafc' }}>
        {detections.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
              style={{ background: darkMode ? '#0a1525' : '#ffffff', border: `1px solid ${borderC}` }}>
              <Ship className="h-8 w-8" style={{ color: labelC }} />
            </div>
            <h3 className="text-lg font-bold mb-1" style={{ color: textC }}>No Targets Found</h3>
            <p className="text-sm max-w-sm" style={{ color: labelC }}>
              Upload a SAR image and execute the detection pipeline to begin tracking ship coordinates.
            </p>
          </div>
        ) : (
          <div className="rounded-xl overflow-hidden shadow-lg border" style={{ background: cardBg, borderColor: borderC }}>
            <table className="w-full relative">
              <thead className="sticky top-0 z-10" style={{ background: headerBg }}>
                <tr>
                  <th className="w-4 border-b" style={{ borderColor: borderC }} />
                  <ColHeader label="Target ID" sk="id" className="w-32" />
                  <ColHeader label="Coordinate Map (Lat / Lon) or Pixel Center" sk="lat" />
                  <ColHeader label="Classification" sk="type" className="w-48" />
                  <ColHeader label="Confidence" sk="confidence" className="w-32" />
                </tr>
              </thead>
              <tbody>
                {sorted.map((det, index) => {
                  const isGeo = det.lat !== null && det.lon !== null;
                  return (
                    <tr key={det.id} className="transition-colors hover:bg-black/5 dark:hover:bg-white/5 border-b last:border-0"
                      style={{ borderColor: borderC }}>
                      <td className="px-4 py-3">
                        <div className="text-xs font-bold" style={{ color: labelC }}>{index + 1}</div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm font-bold font-mono" style={{ color: '#00c8ff' }}>{det.id}</span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <MapPin className="h-4 w-4" style={{ color: isGeo ? '#22c55e' : '#f59e0b' }} />
                          <span className="text-sm font-mono tracking-wide" style={{ color: textC }}>
                            {isGeo
                              ? `${det.lat?.toFixed(5)}°N, ${det.lon?.toFixed(5)}°E`
                              : `Px: ${det.pixelCenter?.[0] ? Math.round(det.pixelCenter[0]) : 0}, ${det.pixelCenter?.[1] ? Math.round(det.pixelCenter[1]) : 0}`
                            }
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="inline-flex items-center px-2 py-1 rounded text-xs font-bold"
                          style={{ background: darkMode ? '#1a2d45' : '#f1f5f9', color: labelC }}>
                          {det.type}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: darkMode ? '#1a2d45' : '#e2e8f0' }}>
                            <div className="h-full rounded-full"
                              style={{
                                width: `${Math.max(0, Math.min(100, det.confidence * 100))}%`,
                                background: det.confidence > 0.85 ? '#22c55e' : det.confidence > 0.6 ? '#f59e0b' : '#ef4444'
                              }} />
                          </div>
                          <span className="text-xs font-mono font-bold w-10 text-right" style={{ color: textC }}>
                            {Math.round(det.confidence * 100)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
