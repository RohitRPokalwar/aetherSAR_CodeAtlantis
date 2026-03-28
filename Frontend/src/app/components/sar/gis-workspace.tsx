import React, { useRef, useState, useCallback } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Image as ImageIcon, Layers } from 'lucide-react';
import { Detection } from './types';

interface GISWorkspaceProps {
  detections: Detection[];
  selectedId: string | null;
  onSelectDetection: (id: string | null) => void;
  imageUrl?: string | null;
  resultImageUrl?: string | null;
}

type ViewMode = 'original' | 'detected' | 'split';

export const GISWorkspace: React.FC<GISWorkspaceProps> = ({
  detections, selectedId, onSelectDetection, imageUrl, resultImageUrl,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [viewMode, setViewMode] = useState<ViewMode>('detected');

  const hasOriginal = !!imageUrl;
  const hasResult = !!resultImageUrl;
  const showImage = viewMode === 'original' ? imageUrl : (resultImageUrl || imageUrl);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom(z => Math.max(0.1, Math.min(10, z * factor)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  const zoomIn = () => setZoom(z => Math.min(10, z * 1.4));
  const zoomOut = () => setZoom(z => Math.max(0.1, z / 1.4));

  return (
    <div ref={containerRef}
      className="relative flex-1 overflow-hidden"
      style={{ background: '#040c18' }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}>

      {/* Image display area */}
      {showImage ? (
        <div
          className="w-full h-full flex items-center justify-center"
          style={{ cursor: isDragging ? 'grabbing' : 'grab' }}>
          <img
            src={showImage}
            alt="SAR Detection Result"
            draggable={false}
            className="select-none"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: isDragging ? 'none' : 'transform 0.15s ease-out',
              imageRendering: zoom > 2 ? 'pixelated' : 'auto',
            }}
          />
        </div>
      ) : (
        /* Empty state placeholder */
        <div className="w-full h-full flex flex-col items-center justify-center">
          <div className="text-center max-w-sm">
            <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
              style={{ background: 'rgba(0,200,255,0.06)', border: '1px solid rgba(0,200,255,0.15)' }}>
              <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="rgba(0,200,255,0.3)" strokeWidth="1" />
                <circle cx="16" cy="16" r="9" stroke="rgba(0,200,255,0.2)" strokeWidth="1" />
                <circle cx="16" cy="16" r="4" stroke="rgba(0,200,255,0.4)" strokeWidth="1" />
                <line x1="16" y1="2" x2="16" y2="30" stroke="rgba(0,200,255,0.2)" strokeWidth="0.5" />
                <line x1="2" y1="16" x2="30" y2="16" stroke="rgba(0,200,255,0.2)" strokeWidth="0.5" />
              </svg>
            </div>
            <p className="text-sm mb-1" style={{ color: '#334155' }}>No SAR imagery loaded</p>
            <p className="text-xs" style={{ color: '#1e3a5f' }}>Upload imagery and run detection to begin</p>
          </div>
        </div>
      )}

      {/* Top-right controls */}
      <div className="absolute right-3 top-3 flex flex-col gap-1.5">
        {[
          { icon: <ZoomIn className="h-4 w-4" />, action: zoomIn, tip: 'Zoom In' },
          { icon: <ZoomOut className="h-4 w-4" />, action: zoomOut, tip: 'Zoom Out' },
          { icon: <Maximize2 className="h-4 w-4" />, action: resetView, tip: 'Fit to View' },
        ].map(({ icon, action, tip }) => (
          <button key={tip} onClick={action} title={tip}
            className="w-8 h-8 flex items-center justify-center rounded transition-all"
            style={{ background: 'rgba(8,18,30,0.9)', border: '1px solid #1a2d45', color: '#475569' }}
            onMouseEnter={e => { e.currentTarget.style.color = '#00c8ff'; e.currentTarget.style.borderColor = 'rgba(0,200,255,0.4)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = '#475569'; e.currentTarget.style.borderColor = '#1a2d45'; }}>
            {icon}
          </button>
        ))}
      </div>

      {/* View mode toggle — only when both images are available */}
      {hasOriginal && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-1 p-1 rounded"
          style={{ background: 'rgba(8,18,30,0.95)', border: '1px solid #1a2d45' }}>
          {([
            { mode: 'original' as ViewMode, label: 'Original', icon: <ImageIcon className="h-3 w-3" /> },
            { mode: 'detected' as ViewMode, label: 'Detected', icon: <Layers className="h-3 w-3" /> },
          ]).map(({ mode, label, icon }) => (
            <button key={mode} onClick={() => setViewMode(mode)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-all"
              style={{
                background: viewMode === mode ? 'rgba(0,200,255,0.15)' : 'transparent',
                color: viewMode === mode ? '#00c8ff' : '#475569',
              }}>
              {icon} {label}
            </button>
          ))}
        </div>
      )}

      {/* Zoom indicator */}
      <div className="absolute left-3 bottom-3 text-xs font-mono px-2 py-1 rounded"
        style={{ background: 'rgba(8,18,30,0.85)', border: '1px solid #1a2d45', color: '#334155' }}>
        {Math.round(zoom * 100)}%
      </div>

      {/* Detection count overlay */}
      {detections.length > 0 && (
        <div className="absolute left-3 top-3 px-3 py-1.5 rounded text-xs"
          style={{ background: 'rgba(8,18,30,0.9)', border: '1px solid rgba(0,200,255,0.2)', color: '#00c8ff' }}>
          {detections.length} detections
        </div>
      )}
    </div>
  );
};