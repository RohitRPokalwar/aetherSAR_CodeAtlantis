import React, { useState } from 'react';
import { Play, Square, Cpu } from 'lucide-react';
import { Detection, DetectionSummary, PipelineConfig, FileMetadata, ProcessingStage } from '../types';
import { GISWorkspace } from '../gis-workspace';
import { DetectionAnalyticsTray } from '../detection-analytics-tray';

interface DetectionPageProps {
  detections: Detection[];
  selectedId: string | null;
  onSelectDetection: (id: string | null) => void;
  config: PipelineConfig;
  onConfigChange: (config: PipelineConfig) => void;
  onRunDetection: () => void;
  isProcessing: boolean;
  hasFile: boolean;
  modelReady: boolean;
  fileId: string | null;
  originalImageUrl: string | null;
  resultImageUrl: string | null;
  summary: DetectionSummary | null;
  onExportCSV: () => void;
  onExportPDF: () => void;
  darkMode: boolean;
}

export const DetectionPage: React.FC<DetectionPageProps> = ({
  detections, selectedId, onSelectDetection,
  config, onConfigChange, onRunDetection,
  isProcessing, hasFile, modelReady,
  fileId, originalImageUrl, resultImageUrl,
  summary, onExportCSV, onExportPDF, darkMode,
}) => {
  const [isAnalyticsTrayOpen, setIsAnalyticsTrayOpen] = useState(detections.length > 0);

  // Open tray automatically when detections arrive
  React.useEffect(() => {
    if (detections.length > 0) setIsAnalyticsTrayOpen(true);
  }, [detections.length]);

  const panelBg = darkMode ? '#08121e' : '#ffffff';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const sectionLabel = darkMode ? '#334155' : '#94a3b8';
  const sliderPct = ((config.confidenceThreshold - 10) / 85) * 100;

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Left control strip */}
      <div
        className="flex-none flex flex-col h-full overflow-hidden"
        style={{ width: '240px', minWidth: '240px', background: panelBg, borderRight: `1px solid ${borderC}` }}
      >
        {/* Model Status */}
        <div className="px-4 py-3 border-b" style={{ borderColor: borderC }}>
          <div className="flex items-center gap-2 p-2.5 rounded"
            style={{
              background: modelReady ? 'rgba(34,197,94,0.06)' : 'rgba(239,68,68,0.06)',
              border: `1px solid ${modelReady ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'}`,
            }}>
            <div className="w-2 h-2 rounded-full" style={{
              background: modelReady ? '#22c55e' : '#ef4444',
              boxShadow: modelReady ? '0 0 6px rgba(34,197,94,0.5)' : '0 0 6px rgba(239,68,68,0.5)',
            }} />
            <span className="text-xs" style={{ color: modelReady ? '#22c55e' : '#ef4444' }}>
              {modelReady ? 'Model ready' : 'Backend offline'}
            </span>
          </div>
        </div>

        {/* Confidence Threshold */}
        <div className="px-4 py-3 border-b" style={{ borderColor: borderC }}>
          <div className="text-xs uppercase tracking-widest mb-3" style={{ color: sectionLabel }}>Detection Settings</div>
          <div className="mb-4">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs" style={{ color: '#64748b' }}>Confidence Threshold</span>
              <span className="text-xs font-mono" style={{ color: '#00c8ff' }}>{config.confidenceThreshold}%</span>
            </div>
            <input
              type="range" min={10} max={95} step={1}
              value={config.confidenceThreshold}
              onChange={e => onConfigChange({ ...config, confidenceThreshold: Number(e.target.value) })}
              className="w-full h-1 rounded-full appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(90deg, #00c8ff ${sliderPct}%, #1a2d45 ${sliderPct}%)`,
                accentColor: '#00c8ff',
              }}
            />
            <div className="text-xs mt-1" style={{ color: darkMode ? '#1e3a5f' : '#cbd5e1' }}>
              Higher values reduce false positives
            </div>
          </div>
        </div>

        {/* Quick info */}
        <div className="px-4 py-3 border-b flex items-center gap-2" style={{ borderColor: borderC }}>
          <div className="flex items-center gap-1 px-2 py-0.5 rounded text-xs"
            style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.15)', color: '#00c8ff' }}>
            <Cpu className="h-3 w-3" /> YOLO-SAR
          </div>
          {detections.length > 0 && (
            <span className="text-xs font-mono" style={{ color: '#22c55e' }}>{detections.length} targets</span>
          )}
        </div>

        <div className="flex-1" />

        {/* Run button */}
        <div className="flex-none p-4 border-t" style={{ borderColor: borderC }}>
          <button
            onClick={onRunDetection}
            disabled={isProcessing || !hasFile}
            className="w-full py-3 rounded flex items-center justify-center gap-2 transition-all duration-200 disabled:opacity-50"
            style={{
              background: isProcessing || !hasFile ? (darkMode ? '#0a1525' : '#e2e8f0') : 'linear-gradient(135deg, #0066cc, #00c8ff)',
              color: '#fff',
              boxShadow: isProcessing || !hasFile ? 'none' : '0 0 20px rgba(0,200,255,0.25)',
            }}>
            {isProcessing ? (
              <><Square className="h-4 w-4" /><span className="text-sm">Processing…</span></>
            ) : !hasFile ? (
              <span className="text-sm">Upload imagery first</span>
            ) : (
              <><Play className="h-4 w-4" /><span className="text-sm">Run Detection</span></>
            )}
          </button>
          <div className="mt-2 text-center text-xs" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>
            Confidence: {config.confidenceThreshold}%
          </div>
        </div>
      </div>

      {/* Main workspace area */}
      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="flex-1 relative overflow-hidden flex">
          <GISWorkspace
            detections={detections}
            selectedId={selectedId}
            onSelectDetection={onSelectDetection}
            imageUrl={originalImageUrl}
            resultImageUrl={resultImageUrl}
          />
        </div>

        <DetectionAnalyticsTray
          detections={detections}
          selectedId={selectedId}
          onSelectDetection={onSelectDetection}
          isOpen={isAnalyticsTrayOpen}
          onToggle={() => setIsAnalyticsTrayOpen(v => !v)}
          summary={summary}
          onExportCSV={onExportCSV}
        />
      </div>
    </div>
  );
};
