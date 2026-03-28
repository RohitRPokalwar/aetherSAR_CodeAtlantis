import React from 'react';
import { ImageIcon, Ship, Radar, Anchor } from 'lucide-react';
import { Detection, DetectionSummary } from '../types';

interface GalleryPageProps {
  detections: Detection[];
  fileId: string | null;
  darkMode: boolean;
  summary: DetectionSummary | null;
}

const MAX_GALLERY_ITEMS = 7;

export const GalleryPage: React.FC<GalleryPageProps> = ({
  detections, fileId, darkMode, summary,
}) => {
  const bg = darkMode ? '#040c18' : '#f8fafc';
  const panelBg = darkMode ? '#08121e' : '#ffffff';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const textC = darkMode ? '#e2e8f0' : '#1e293b';
  const labelC = darkMode ? '#94a3b8' : '#64748b';
  const cardBg = darkMode ? '#0a1525' : '#ffffff';
  const accentC = '#00c8ff';

  // Top detections by confidence
  const topDetections = [...detections]
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, MAX_GALLERY_ITEMS);

  /* ── Empty state ── */
  if (!fileId || detections.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center" style={{ background: bg }}>
        <div className="text-center max-w-md px-6">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6"
            style={{
              background: `rgba(0,200,255,0.06)`,
              border: `1px solid rgba(0,200,255,0.15)`,
            }}
          >
            <ImageIcon className="w-9 h-9" style={{ color: 'rgba(0,200,255,0.4)' }} />
          </div>
          <h2 className="text-xl font-bold mb-2" style={{ color: textC }}>
            No Detection Results
          </h2>
          <p className="text-sm" style={{ color: labelC }}>
            Upload SAR imagery and run detection to see YOLO model output images here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden" style={{ background: bg }}>
      {/* Header */}
      <div
        className="flex-none flex items-center justify-between px-6 py-4 border-b"
        style={{ borderColor: borderC, background: panelBg }}
      >
        <div className="flex items-center gap-3">
          <div
            className="p-2.5 rounded-lg"
            style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.15)' }}
          >
            <Radar className="h-5 w-5" style={{ color: accentC }} />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight" style={{ color: textC }}>
              YOLO Detection Output Gallery
            </h1>
            <p className="text-xs mt-0.5" style={{ color: labelC }}>
              Top {topDetections.length} ship detections ranked by confidence
              {summary && ` · ${summary.totalDetections} total across ${summary.tilesProcessed} tiles`}
            </p>
          </div>
        </div>

        {/* Summary pills */}
        <div className="flex items-center gap-2">
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-mono"
            style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)', color: '#22c55e' }}
          >
            <Ship className="h-3.5 w-3.5" />
            {detections.length} ships detected
          </div>
          {summary && (
            <div
              className="px-3 py-1.5 rounded-full text-xs font-mono"
              style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.2)', color: accentC }}
            >
              {(summary.processingTimeMs / 1000).toFixed(1)}s inference
            </div>
          )}
        </div>
      </div>

      {/* Gallery Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 max-w-[1600px] mx-auto">
          {topDetections.map((det, idx) => {
            const confPct = Math.round(det.confidence * 100);
            const confColor = confPct >= 80 ? '#22c55e' : confPct >= 60 ? '#eab308' : '#f97316';

            return (
              <div
                key={det.id}
                className="group flex flex-col rounded-xl overflow-hidden transition-all duration-300 hover:-translate-y-1"
                style={{
                  background: cardBg,
                  border: `1px solid ${borderC}`,
                  boxShadow: `0 4px 20px rgba(0,0,0,${darkMode ? '0.3' : '0.08'})`,
                }}
              >
                {/* Image */}
                <div
                  className="relative aspect-square overflow-hidden"
                  style={{ background: '#000' }}
                >
                  <img
                    src={`/api/crop/${fileId}/${det.id}`}
                    alt={`Ship detection ${det.id}`}
                    className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
                    loading="lazy"
                  />

                  {/* Rank badge */}
                  <div
                    className="absolute top-3 left-3 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                    style={{
                      background: 'linear-gradient(135deg, #0066cc, #00c8ff)',
                      color: '#fff',
                      boxShadow: '0 2px 8px rgba(0,200,255,0.4)',
                    }}
                  >
                    #{idx + 1}
                  </div>

                  {/* Confidence badge */}
                  <div
                    className="absolute top-3 right-3 px-2.5 py-1 rounded-lg text-xs font-mono font-bold backdrop-blur-sm"
                    style={{
                      background: 'rgba(0,0,0,0.7)',
                      color: confColor,
                      border: `1px solid ${confColor}40`,
                    }}
                  >
                    {confPct}%
                  </div>

                  {/* Ship type badge */}
                  <div
                    className="absolute bottom-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold backdrop-blur-sm"
                    style={{
                      background: 'rgba(0,0,0,0.75)',
                      color: '#fff',
                      border: '1px solid rgba(255,255,255,0.15)',
                    }}
                  >
                    <Anchor className="h-3 w-3" style={{ color: accentC }} />
                    {det.type}
                  </div>
                </div>

                {/* Details card */}
                <div className="px-4 py-3 space-y-2" style={{ borderTop: `1px solid ${borderC}` }}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold font-mono" style={{ color: textC }}>
                      {det.id}
                    </span>
                    <span
                      className="text-xs px-2 py-0.5 rounded-full font-bold"
                      style={{
                        background: det.ais ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                        color: det.ais ? '#22c55e' : '#ef4444',
                        border: `1px solid ${det.ais ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                      }}
                    >
                      {det.ais ? 'AIS' : 'DARK'}
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span style={{ color: labelC }}>Confidence</span>
                      <span className="font-mono font-bold" style={{ color: confColor }}>{confPct}%</span>
                    </div>
                    <div className="h-1.5 rounded-full overflow-hidden" style={{ background: darkMode ? '#1a2d45' : '#e2e8f0' }}>
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${confPct}%`,
                          background: `linear-gradient(90deg, ${confColor}, ${confColor}cc)`,
                          boxShadow: `0 0 6px ${confColor}40`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Metadata rows */}
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 pt-1">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>Length</span>
                      <span className="font-mono" style={{ color: textC }}>{det.lengthM}m</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>Beam</span>
                      <span className="font-mono" style={{ color: textC }}>{det.beamM}m</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>Heading</span>
                      <span className="font-mono" style={{ color: textC }}>{det.headingDeg}°</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>RCS</span>
                      <span className="font-mono" style={{ color: textC }}>{det.rcs} dBsm</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer note */}
        {detections.length > MAX_GALLERY_ITEMS && (
          <div className="mt-8 text-center text-xs" style={{ color: labelC }}>
            Showing top {MAX_GALLERY_ITEMS} of {detections.length} total detections
          </div>
        )}
      </div>
    </div>
  );
};
