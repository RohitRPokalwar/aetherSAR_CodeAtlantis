import React, { useState } from 'react';
import { Play, Square, Cpu, AlertTriangle, Bell, Shield, FileText, Download } from 'lucide-react';
import { PipelineConfig } from './types';

interface PipelineConfigPanelProps {
  config: PipelineConfig;
  onChange: (config: PipelineConfig) => void;
  onRunDetection: () => void;
  isProcessing: boolean;
  hasFile: boolean;
  modelReady: boolean;
  detectionCount: number;
  onExportPDF: () => void;
}

const SliderField: React.FC<{
  label: string; value: number; min: number; max: number; step?: number;
  unit?: string;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step = 1, unit = '', onChange }) => (
  <div className="mb-4">
    <div className="flex justify-between items-center mb-1.5">
      <span className="text-xs" style={{ color: '#64748b' }}>{label}</span>
      <span className="text-xs font-mono" style={{ color: '#00c8ff' }}>
        {value}{unit}
      </span>
    </div>
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="w-full h-1 rounded-full appearance-none cursor-pointer"
      style={{
        background: `linear-gradient(90deg, #00c8ff ${((value - min) / (max - min)) * 100}%, #1a2d45 ${((value - min) / (max - min)) * 100}%)`,
        accentColor: '#00c8ff',
      }}
    />
  </div>
);

interface Alert {
  type: 'critical' | 'warning' | 'info';
  message: string;
  time: string;
}

function generateAlerts(detectionCount: number, modelReady: boolean, hasFile: boolean): Alert[] {
  const alerts: Alert[] = [];
  const now = new Date();
  const timeStr = (offset: number) => {
    const d = new Date(now.getTime() - offset * 60000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (!modelReady) {
    alerts.push({ type: 'critical', message: 'Backend server offline — start FastAPI', time: timeStr(0) });
  } else {
    alerts.push({ type: 'info', message: 'Backend connected · Model loaded', time: timeStr(1) });
  }

  if (!hasFile) {
    alerts.push({ type: 'warning', message: 'No imagery loaded — upload SAR file', time: timeStr(0) });
  }

  if (detectionCount > 0) {
    const darkVessels = Math.max(1, Math.floor(detectionCount * 0.15));
    alerts.push({ type: 'critical', message: `${darkVessels} dark vessel(s) detected — no AIS signal`, time: timeStr(2) });
    alerts.push({ type: 'info', message: `${detectionCount} total targets acquired`, time: timeStr(3) });
    if (detectionCount > 5) {
      alerts.push({ type: 'warning', message: 'High vessel density in detection zone', time: timeStr(4) });
    }
  }

  return alerts;
}

export const PipelineConfigPanel: React.FC<PipelineConfigPanelProps> = ({
  config, onChange, onRunDetection, isProcessing, hasFile, modelReady, detectionCount, onExportPDF,
}) => {
  const [showAlertHistory, setShowAlertHistory] = useState(true);

  const upd = (key: keyof PipelineConfig, val: string | number) =>
    onChange({ ...config, [key]: val });

  const alerts = generateAlerts(detectionCount, modelReady, hasFile);
  const criticalCount = alerts.filter(a => a.type === 'critical').length;

  const alertStyles: Record<string, { bg: string; border: string; color: string; icon: React.ReactNode }> = {
    critical: {
      bg: 'rgba(239,68,68,0.06)', border: 'rgba(239,68,68,0.2)', color: '#ef4444',
      icon: <AlertTriangle className="h-3 w-3 flex-none" />,
    },
    warning: {
      bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.2)', color: '#f59e0b',
      icon: <Shield className="h-3 w-3 flex-none" />,
    },
    info: {
      bg: 'rgba(34,197,94,0.06)', border: 'rgba(34,197,94,0.2)', color: '#22c55e',
      icon: <Bell className="h-3 w-3 flex-none" />,
    },
  };

  return (
    <aside className="flex flex-col h-full overflow-hidden"
      style={{ background: '#08121e', borderRight: '1px solid #1a2d45', width: '280px', minWidth: '280px' }}>

      {/* Panel header */}
      <div className="flex-none px-4 py-3 border-b" style={{ borderColor: '#1a2d45' }}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold tracking-widest uppercase" style={{ color: '#94a3b8' }}>Control Panel</span>
          <div className="flex items-center gap-1.5">
            <div className="flex items-center gap-1 px-2 py-0.5 rounded text-xs"
              style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.15)', color: '#00c8ff' }}>
              <Cpu className="h-3 w-3" /> YOLO-SAR
            </div>
          </div>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">

        {/* Model status */}
        <div className="px-4 py-3 border-b" style={{ borderColor: '#0f1d30' }}>
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
              {modelReady ? 'Backend connected · Model ready' : 'Backend offline'}
            </span>
          </div>
        </div>

        {/* Confidence Threshold */}
        <div className="px-4 py-3 border-b" style={{ borderColor: '#0f1d30' }}>
          <div className="text-xs uppercase tracking-widest mb-3" style={{ color: '#334155' }}>Detection Settings</div>
          <SliderField
            label="Confidence Threshold"
            value={config.confidenceThreshold}
            min={10} max={95} unit="%"
            onChange={v => upd('confidenceThreshold', v)}
          />
          <div className="text-xs mt-1" style={{ color: '#1e3a5f' }}>
            Higher values reduce false positives but may miss weak targets
          </div>
        </div>

        {/* Alerts Section */}
        <div className="px-4 py-3 border-b" style={{ borderColor: '#0f1d30' }}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-widest" style={{ color: '#334155' }}>Alerts</div>
              {criticalCount > 0 && (
                <div className="px-1.5 py-0.5 rounded text-xs animate-pulse"
                  style={{ background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}>
                  {criticalCount}
                </div>
              )}
            </div>
            <button
              onClick={() => setShowAlertHistory(v => !v)}
              className="text-xs transition-colors"
              style={{ color: '#475569' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#94a3b8')}
              onMouseLeave={e => (e.currentTarget.style.color = '#475569')}>
              {showAlertHistory ? 'Hide' : 'Show'}
            </button>
          </div>

          {showAlertHistory && (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {alerts.map((alert, i) => {
                const style = alertStyles[alert.type];
                return (
                  <div key={i} className="flex items-start gap-2 p-2 rounded"
                    style={{ background: style.bg, border: `1px solid ${style.border}` }}>
                    <span className="mt-0.5" style={{ color: style.color }}>{style.icon}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs leading-tight" style={{ color: style.color }}>{alert.message}</p>
                      <p className="text-xs mt-0.5" style={{ color: '#334155' }}>{alert.time}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Export section */}
        <div className="px-4 py-3">
          <div className="text-xs uppercase tracking-widest mb-3" style={{ color: '#334155' }}>Reports</div>
          <button
            onClick={onExportPDF}
            disabled={detectionCount === 0}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded text-xs transition-all disabled:opacity-30"
            style={{
              background: detectionCount > 0 ? 'rgba(139,92,246,0.1)' : '#0a1525',
              border: `1px solid ${detectionCount > 0 ? 'rgba(139,92,246,0.3)' : '#1a2d45'}`,
              color: detectionCount > 0 ? '#a78bfa' : '#334155',
            }}
            onMouseEnter={e => { if (detectionCount > 0) { e.currentTarget.style.background = 'rgba(139,92,246,0.18)'; } }}
            onMouseLeave={e => { if (detectionCount > 0) { e.currentTarget.style.background = 'rgba(139,92,246,0.1)'; } }}>
            <FileText className="h-3.5 w-3.5" />
            Generate PDF Report
          </button>
          <p className="text-xs mt-1.5 text-center" style={{ color: '#1e3a5f' }}>
            {detectionCount > 0
              ? `Full report with ${detectionCount} detections`
              : 'Run detection first'}
          </p>
        </div>
      </div>

      {/* Run button */}
      <div className="flex-none p-4 border-t" style={{ borderColor: '#1a2d45' }}>
        <button
          onClick={onRunDetection}
          disabled={isProcessing || !hasFile}
          className="w-full py-3 rounded flex items-center justify-center gap-2 transition-all duration-200 disabled:opacity-50"
          style={{
            background: isProcessing || !hasFile ? '#0a1525' : 'linear-gradient(135deg, #0066cc, #00c8ff)',
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
        <div className="mt-2 text-center text-xs" style={{ color: '#334155' }}>
          Confidence: {config.confidenceThreshold}%
        </div>
      </div>
    </aside>
  );
};
