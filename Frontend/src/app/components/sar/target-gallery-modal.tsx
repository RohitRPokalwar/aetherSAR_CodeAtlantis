import React, { useState } from 'react';
import { X, Target } from 'lucide-react';
import { Detection } from './types';

interface TargetGalleryModalProps {
  isOpen: boolean;
  onClose: () => void;
  fileId: string | null;
  detections: Detection[];
  darkMode: boolean;
}

export const TargetGalleryModal: React.FC<TargetGalleryModalProps> = ({
  isOpen, onClose, fileId, detections, darkMode
}) => {
  const [loadLimit, setLoadLimit] = useState(20);

  if (!isOpen || !fileId || detections.length === 0) return null;

  const bg = darkMode ? 'rgba(4,12,24,0.95)' : 'rgba(248,250,252,0.95)';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const textC = darkMode ? '#e2e8f0' : '#1e293b';
  const labelC = darkMode ? '#94a3b8' : '#64748b';
  const cardBg = darkMode ? '#0a1525' : '#ffffff';

  const sortedDetections = [...detections]
    .sort((a, b) => b.confidence - a.confidence);
  
  const visibleDetections = sortedDetections.slice(0, loadLimit);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
      <div 
        className="relative w-full max-w-6xl max-h-[90vh] flex flex-col rounded-xl overflow-hidden shadow-2xl"
        style={{ background: bg, border: `1px solid ${borderC}` }}
      >
        {/* Header */}
        <div className="flex-none flex items-center justify-between p-4 border-b" style={{ borderColor: borderC, background: darkMode ? '#060e1a' : '#f1f5f9' }}>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg" style={{ background: 'rgba(0,200,255,0.1)' }}>
              <Target className="h-5 w-5" style={{ color: '#00c8ff' }} />
            </div>
            <div>
              <h2 className="text-lg font-bold" style={{ color: textC }}>High-Fidelity Tile Gallery</h2>
              <p className="text-xs" style={{ color: labelC }}>Showing {visibleDetections.length} of {detections.length} neighborhood tiles prioritized by primary target confidence</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded transition-colors hover:bg-red-500/10 hover:text-red-500"
            style={{ color: labelC }}
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Target Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {visibleDetections.map((det, idx) => (
              <div 
                key={det.id}
                className="flex flex-col rounded-lg overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-lg"
                style={{ background: cardBg, border: `1px solid ${borderC}`, boxShadow: `0 4px 12px rgba(0,0,0,${darkMode ? '0.2' : '0.05'})` }}
              >
                {/* Image Crop */}
                <div className="relative aspect-square bg-black flex items-center justify-center overflow-hidden border-b" style={{ borderColor: borderC }}>
                  <img 
                    src={`/api/crop/${fileId}/${det.id}`}
                    alt={`Detection ${det.id}`}
                    className="w-full h-full object-contain"
                    loading="lazy"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-bold" style={{ background: 'rgba(0,0,0,0.7)', color: '#fff', border: '1px solid rgba(255,255,255,0.2)' }}>
                    Tile #{idx + 1}
                  </div>
                  <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs font-mono font-bold" style={{ background: 'rgba(0,0,0,0.7)', color: '#00c8ff', border: '1px solid rgba(0,200,255,0.3)' }}>
                    {Math.round(det.confidence * 100)}%
                  </div>
                </div>

                {/* Details */}
                <div className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-bold font-mono" style={{ color: textC }}>Tile #{idx + 1}</span>
                    <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: darkMode ? '#1a2d45' : '#f1f5f9', color: labelC }}>{det.type}</span>
                  </div>
                  
                  <div className="space-y-1.5 mt-3">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>Size:</span>
                      <span className="font-mono" style={{ color: textC }}>{det.lengthM}m × {det.beamM}m</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>Heading:</span>
                      <span className="font-mono" style={{ color: textC }}>{det.headingDeg}°</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span style={{ color: labelC }}>AIS:</span>
                      <span className="font-mono font-bold" style={{ color: det.ais ? '#22c55e' : '#ef4444' }}>{det.ais || 'DARK'}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {loadLimit < detections.length && (
            <div className="mt-8 flex justify-center">
              <button
                onClick={() => setLoadLimit(l => l + 20)}
                className="px-6 py-2 rounded-lg text-sm font-bold transition-all"
                style={{
                  background: 'linear-gradient(135deg, #00c8ff, #0066cc)',
                  color: '#fff',
                  boxShadow: '0 0 15px rgba(0,200,255,0.3)',
                }}
              >
                Load More Detections
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
