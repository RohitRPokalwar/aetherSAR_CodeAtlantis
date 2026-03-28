import React, { useState, useRef, useEffect } from 'react';
import { Upload, Bell, ChevronDown, Hexagon, Satellite, Activity, Sun, Moon, User, LogOut } from 'lucide-react';
import { ProcessingStage, FileMetadata } from './types';

interface TopNavbarProps {
  processingStage: ProcessingStage;
  onOpenUpload: () => void;
  detectionCount: number;
  fileMetadata: FileMetadata | null;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  alerts: { type: string; message: string; time: string }[];
}

export const TopNavbar: React.FC<TopNavbarProps> = ({
  processingStage, onOpenUpload, detectionCount, fileMetadata, darkMode, onToggleDarkMode, alerts,
}) => {
  const isProcessing = processingStage > 0 && processingStage < 4;
  const [alertOpen, setAlertOpen] = useState(false);
  const [userOpen, setUserOpen] = useState(false);
  const alertRef = useRef<HTMLDivElement>(null);
  const userRef = useRef<HTMLDivElement>(null);

  // Close dropdowns on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (alertRef.current && !alertRef.current.contains(e.target as Node)) setAlertOpen(false);
      if (userRef.current && !userRef.current.contains(e.target as Node)) setUserOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const criticalCount = alerts.filter(a => a.type === 'critical').length;

  const bg = darkMode ? '#080f1c' : '#f8fafc';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#475569';
  const textMuted = darkMode ? '#64748b' : '#64748b';
  const dropBg = darkMode ? '#0c1829' : '#ffffff';
  const dropBorder = darkMode ? '#1a2d45' : '#e2e8f0';
  const hoverBg = darkMode ? '#0f1d30' : '#f1f5f9';

  return (
    <header className="flex-none flex items-center justify-between px-4 h-12 border-b"
      style={{ background: bg, borderColor: borderC }}>

      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Hexagon className="h-5 w-5" style={{ color: '#00c8ff' }} fill="rgba(0,200,255,0.15)" />
          <span className="text-sm font-black tracking-widest" style={{ color: textPrimary }}>aetherSAR</span>
        </div>
        <div className="w-px h-5" style={{ background: borderC }} />
        <span className="text-xs tracking-widest uppercase" style={{ color: textMuted }}>Maritime Domain Awareness</span>
      </div>

      {/* Mission Info */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-xs" style={{ color: textSecondary }}>
          <div className="flex items-center gap-1.5">
            <span style={{ color: textMuted }}>MSN</span>
            <span>{new Date().toISOString().slice(0, 10).replace(/-/g, '')}</span>
          </div>
          <div className="w-px h-4" style={{ background: borderC }} />
          <div className="flex items-center gap-1.5">
            <Satellite className="h-3 w-3" style={{ color: textSecondary }} />
            <span>{fileMetadata ? fileMetadata.filename : 'SAR-X Band · 1m GSD'}</span>
          </div>
          {fileMetadata && (
            <>
              <div className="w-px h-4" style={{ background: borderC }} />
              <div className="flex items-center gap-1.5">
                <span style={{ color: textMuted }}>DIM</span>
                <span>{fileMetadata.width}×{fileMetadata.height}</span>
              </div>
            </>
          )}
        </div>

        {isProcessing && (
          <div className="flex items-center gap-2 px-3 py-1 rounded text-xs"
            style={{ background: 'rgba(0,200,255,0.08)', border: '1px solid rgba(0,200,255,0.2)' }}>
            <Activity className="h-3 w-3 animate-pulse" style={{ color: '#00c8ff' }} />
            <span style={{ color: '#00c8ff' }}>PROCESSING</span>
          </div>
        )}
        {processingStage === 4 && (
          <div className="flex items-center gap-2 px-3 py-1 rounded text-xs"
            style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#22c55e' }} />
            <span style={{ color: '#22c55e' }}>{detectionCount} TARGETS ACQUIRED</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {/* Upload button */}
        <button onClick={onOpenUpload}
          className="flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-all duration-200"
          style={{ background: 'rgba(0,200,255,0.1)', border: '1px solid rgba(0,200,255,0.25)', color: '#00c8ff' }}
          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,200,255,0.18)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'rgba(0,200,255,0.1)')}>
          <Upload className="h-3.5 w-3.5" />
          Upload Imagery
        </button>

        {/* Alert Bell — clickable dropdown */}
        <div ref={alertRef} className="relative">
          <button onClick={() => { setAlertOpen(v => !v); setUserOpen(false); }}
            className="p-2 rounded transition-colors relative"
            style={{ color: alertOpen ? '#00c8ff' : textSecondary }}
            onMouseEnter={e => (e.currentTarget.style.color = '#94a3b8')}
            onMouseLeave={e => { if (!alertOpen) e.currentTarget.style.color = textSecondary; }}>
            <Bell className="h-4 w-4" />
            {criticalCount > 0 && (
              <div className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 rounded-full flex items-center justify-center text-[9px] font-bold animate-pulse"
                style={{ background: '#ef4444', color: '#fff' }}>
                {criticalCount}
              </div>
            )}
          </button>

          {alertOpen && (
            <div className="absolute right-0 top-full mt-1 w-72 rounded-lg shadow-xl z-50 overflow-hidden"
              style={{ background: dropBg, border: `1px solid ${dropBorder}` }}>
              <div className="px-3 py-2 border-b" style={{ borderColor: dropBorder }}>
                <span className="text-xs font-bold uppercase tracking-widest" style={{ color: textPrimary }}>
                  System Alerts
                </span>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {alerts.length === 0 ? (
                  <div className="px-3 py-4 text-center text-xs" style={{ color: textMuted }}>No alerts</div>
                ) : (
                  alerts.map((alert, i) => {
                    const colors: Record<string, string> = { critical: '#ef4444', warning: '#f59e0b', info: '#22c55e' };
                    const c = colors[alert.type] || '#475569';
                    return (
                      <div key={i} className="px-3 py-2.5 border-b flex items-start gap-2"
                        style={{ borderColor: dropBorder }}>
                        <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-none" style={{ background: c }} />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs" style={{ color: c }}>{alert.message}</p>
                          <p className="text-[10px] mt-0.5" style={{ color: textMuted }}>{alert.time}</p>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>

        {/* User "A" icon — clickable with dark/light mode toggle */}
        <div ref={userRef} className="relative">
          <button onClick={() => { setUserOpen(v => !v); setAlertOpen(false); }}
            className="flex items-center gap-2 pl-2 pr-1 py-1 rounded transition-all"
            style={{ border: `1px solid ${userOpen ? 'rgba(0,200,255,0.3)' : borderC}` }}>
            <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
              style={{ background: 'linear-gradient(135deg, #0066cc, #00c8ff)', color: '#fff' }}>A</div>
            <ChevronDown className="h-3 w-3" style={{
              color: textSecondary,
              transform: userOpen ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
            }} />
          </button>

          {userOpen && (
            <div className="absolute right-0 top-full mt-1 w-52 rounded-lg shadow-xl z-50 overflow-hidden"
              style={{ background: dropBg, border: `1px solid ${dropBorder}` }}>
              <div className="px-3 py-2.5 border-b flex items-center gap-2" style={{ borderColor: dropBorder }}>
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: 'linear-gradient(135deg, #0066cc, #00c8ff)', color: '#fff' }}>A</div>
                <div>
                  <p className="text-xs font-medium" style={{ color: textPrimary }}>Admin</p>
                  <p className="text-[10px]" style={{ color: textMuted }}>Operator</p>
                </div>
              </div>

              {/* Dark / Light mode toggle */}
              <button onClick={onToggleDarkMode}
                className="w-full px-3 py-2.5 flex items-center gap-2.5 text-xs transition-colors"
                style={{ color: textPrimary }}
                onMouseEnter={e => (e.currentTarget.style.background = hoverBg)}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                {darkMode ? <Sun className="h-3.5 w-3.5" style={{ color: '#f59e0b' }} /> : <Moon className="h-3.5 w-3.5" style={{ color: '#6366f1' }} />}
                {darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              </button>

              <button onClick={onOpenUpload}
                className="w-full px-3 py-2.5 flex items-center gap-2.5 text-xs transition-colors border-t"
                style={{ color: textPrimary, borderColor: dropBorder }}
                onMouseEnter={e => (e.currentTarget.style.background = hoverBg)}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <Upload className="h-3.5 w-3.5" style={{ color: '#00c8ff' }} />
                Upload New Image
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
