import React, { useMemo } from 'react';
import { AlertTriangle, Shield, Bell, Ship, Radar, MapPin, Eye } from 'lucide-react';
import { Detection } from '../types';
import Particles from '../Particles';

interface AlertsPageProps {
  detections: Detection[];
  modelReady: boolean;
  hasFile: boolean;
  darkMode: boolean;
}

interface SmartAlert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  detail: string;
  time: string;
}

/* ── Quadrant Analysis ── */
function getQuadrantCounts(dets: Detection[]): { name: string; count: number }[] {
  if (dets.length === 0) return [];
  const midLat = dets.reduce((s, d) => s + d.lat, 0) / dets.length;
  const midLon = dets.reduce((s, d) => s + d.lon, 0) / dets.length;
  const q = { NE: 0, NW: 0, SE: 0, SW: 0 };
  dets.forEach(d => {
    const ns = d.lat >= midLat ? 'N' : 'S';
    const ew = d.lon >= midLon ? 'E' : 'W';
    q[(ns + ew) as keyof typeof q]++;
  });
  return Object.entries(q).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);
}

/* ── Cluster Detection (simple distance-based) ── */
function detectClusters(dets: Detection[]): { count: number; center: { lat: number; lon: number } }[] {
  if (dets.length < 3) return [];
  const threshold = 0.05; // ~5km
  const visited = new Set<number>();
  const clusters: { count: number; center: { lat: number; lon: number } }[] = [];

  for (let i = 0; i < dets.length; i++) {
    if (visited.has(i)) continue;
    const cluster = [i];
    visited.add(i);
    for (let j = i + 1; j < dets.length; j++) {
      if (visited.has(j)) continue;
      const dist = Math.sqrt((dets[i].lat - dets[j].lat) ** 2 + (dets[i].lon - dets[j].lon) ** 2);
      if (dist < threshold) { cluster.push(j); visited.add(j); }
    }
    if (cluster.length >= 3) {
      const lats = cluster.map(idx => dets[idx].lat);
      const lons = cluster.map(idx => dets[idx].lon);
      clusters.push({
        count: cluster.length,
        center: {
          lat: lats.reduce((s, v) => s + v, 0) / lats.length,
          lon: lons.reduce((s, v) => s + v, 0) / lons.length,
        },
      });
    }
  }
  return clusters;
}

/* ── Generate Smart Alerts ── */
function generateSmartAlerts(dets: Detection[], modelReady: boolean, hasFile: boolean): SmartAlert[] {
  const alerts: SmartAlert[] = [];
  const now = new Date();
  const t = (offset: number) => {
    const d = new Date(now.getTime() - offset * 60000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  let id = 0;

  // System status
  if (!modelReady) {
    alerts.push({ id: String(id++), severity: 'critical', title: 'Backend Offline', detail: 'FastAPI backend is unreachable — start the server', time: t(0) });
  }
  if (!hasFile) {
    alerts.push({ id: String(id++), severity: 'warning', title: 'No Imagery Loaded', detail: 'Upload a SAR image to begin detection', time: t(0) });
  }

  if (dets.length === 0) return alerts;

  // Dark vessels
  const darkCount = dets.filter(d => !d.ais).length;
  if (darkCount > 0) {
    alerts.push({ id: String(id++), severity: 'critical', title: `${darkCount} Dark Vessel(s) Detected`, detail: 'Ships operating without AIS transponder — possible illicit activity', time: t(2) });
  }

  // High density
  if (dets.length > 10) {
    alerts.push({ id: String(id++), severity: 'warning', title: 'High Vessel Density', detail: `${dets.length} ships detected in single scene — abnormal concentration`, time: t(3) });
  }

  // Cluster detection
  const clusters = detectClusters(dets);
  clusters.forEach((cl, i) => {
    alerts.push({
      id: String(id++), severity: 'warning', title: `Fleet Cluster Detected`,
      detail: `${cl.count} ships grouped near ${cl.center.lat.toFixed(3)}°N, ${cl.center.lon.toFixed(3)}°E`,
      time: t(4 + i),
    });
  });

  // Quadrant density
  const quads = getQuadrantCounts(dets);
  const densest = quads[0];
  if (densest && densest.count >= 3) {
    alerts.push({ id: String(id++), severity: 'info', title: `${densest.name} Quadrant Hotspot`, detail: `${densest.count} ships concentrated in ${densest.name} sector`, time: t(5) });
  }

  // Large vessel
  const largeVessels = dets.filter(d => d.lengthM > 200);
  if (largeVessels.length > 0) {
    alerts.push({ id: String(id++), severity: 'info', title: `${largeVessels.length} Large Vessel(s)`, detail: 'Ships exceeding 200m detected — likely cargo/tanker class', time: t(6) });
  }

  // Target count
  alerts.push({ id: String(id++), severity: 'info', title: `${dets.length} Total Targets Acquired`, detail: 'All detection targets logged for analysis', time: t(7) });

  return alerts;
}

/* ── Alert Item Component ── */
const severityConfig = {
  critical: { color: '#ef4444', bg: 'rgba(239,68,68,0.06)', border: 'rgba(239,68,68,0.2)', icon: <AlertTriangle className="h-4 w-4" /> },
  warning: { color: '#f59e0b', bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.2)', icon: <Shield className="h-4 w-4" /> },
  info: { color: '#22c55e', bg: 'rgba(34,197,94,0.06)', border: 'rgba(34,197,94,0.2)', icon: <Bell className="h-4 w-4" /> },
};

/* ── Main Alerts Page ── */
export const AlertsPage: React.FC<AlertsPageProps> = ({ detections, modelReady, hasFile, darkMode }) => {
  const alerts = useMemo(() => generateSmartAlerts(detections, modelReady, hasFile), [detections, modelReady, hasFile]);
  const quadrants = useMemo(() => getQuadrantCounts(detections), [detections]);
  const clusters = useMemo(() => detectClusters(detections), [detections]);

  const criticals = alerts.filter(a => a.severity === 'critical');
  const warnings = alerts.filter(a => a.severity === 'warning');
  const infos = alerts.filter(a => a.severity === 'info');

  const cardBg = darkMode ? 'rgba(10,21,37,0.65)' : 'rgba(255,255,255,0.75)';
  const borderC = darkMode ? '#1a2d45' : '#e2e8f0';
  const labelColor = darkMode ? '#475569' : '#94a3b8';
  const textColor = darkMode ? '#e2e8f0' : '#1e293b';

  return (
    <div className="flex-1 overflow-hidden relative" style={{ background: darkMode ? '#040c18' : '#f8fafc' }}>

      {/* Particle Background */}
      <Particles
        particleColors={['#ef4444', '#f59e0b', '#00c8ff']}
        particleCount={100}
        particleSpread={10}
        speed={0.04}
        particleBaseSize={150}
        alphaParticles
        sizeRandomness={1}
        cameraDistance={28}
        disableRotation={false}
      />

      <div className="relative z-10 overflow-y-auto h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold tracking-wide" style={{ color: textColor }}>Threat Intelligence</h2>
          <p className="text-xs mt-1" style={{ color: labelColor }}>Real-time maritime threat assessment and anomaly detection</p>
        </div>
        <div className="flex items-center gap-3">
          {criticals.length > 0 && (
            <div className="px-3 py-1 rounded text-xs animate-pulse" style={{ background: 'rgba(239,68,68,0.12)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }}>
              {criticals.length} Critical
            </div>
          )}
          {warnings.length > 0 && (
            <div className="px-3 py-1 rounded text-xs" style={{ background: 'rgba(245,158,11,0.08)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.2)' }}>
              {warnings.length} Warnings
            </div>
          )}
          <div className="px-3 py-1 rounded text-xs" style={{ background: 'rgba(34,197,94,0.08)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.2)' }}>
            {alerts.length} Total
          </div>
        </div>
      </div>

      {/* Alert Feed */}
      <div className="rounded-lg overflow-hidden" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
        <div className="px-4 py-3 border-b" style={{ borderColor: borderC }}>
          <span className="text-xs font-bold uppercase tracking-widest" style={{ color: labelColor }}>Live Alert Feed</span>
        </div>
        <div className="max-h-[400px] overflow-y-auto divide-y" style={{ borderColor: borderC }}>
          {alerts.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Radar className="h-8 w-8 mx-auto mb-2" style={{ color: darkMode ? '#1e3a5f' : '#cbd5e1' }} />
              <p className="text-xs" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>No alerts — upload imagery and run detection</p>
            </div>
          ) : (
            alerts.map(alert => {
              const cfg = severityConfig[alert.severity];
              return (
                <div key={alert.id} className="px-4 py-3 flex items-start gap-3 transition-colors"
                  style={{ borderColor: borderC }}
                  onMouseEnter={e => { e.currentTarget.style.background = cfg.bg; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-none mt-0.5"
                    style={{ background: cfg.bg, color: cfg.color }}>{cfg.icon}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium" style={{ color: cfg.color }}>{alert.title}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider"
                        style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
                        {alert.severity}
                      </span>
                    </div>
                    <p className="text-xs mt-0.5" style={{ color: labelColor }}>{alert.detail}</p>
                  </div>
                  <span className="text-xs flex-none font-mono" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>{alert.time}</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Analysis Cards */}
      <div className="grid grid-cols-2 gap-4">
        {/* Quadrant Density */}
        <div className="rounded-lg p-4" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
          <div className="flex items-center gap-2 mb-4">
            <MapPin className="h-4 w-4" style={{ color: '#00c8ff' }} />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: labelColor }}>Quadrant Density</span>
          </div>
          {quadrants.length === 0 ? (
            <p className="text-xs" style={{ color: darkMode ? '#334155' : '#cbd5e1' }}>No spatial data available</p>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {quadrants.map(q => (
                <div key={q.name} className="p-3 rounded"
                  style={{ background: darkMode ? '#060e1a' : '#f8fafc', border: `1px solid ${borderC}` }}>
                  <div className="text-xs mb-1" style={{ color: labelColor }}>{q.name} Sector</div>
                  <div className="text-xl font-bold font-mono" style={{ color: '#00c8ff' }}>{q.count}</div>
                  <div className="text-[10px]" style={{ color: darkMode ? '#1e3a5f' : '#cbd5e1' }}>ships detected</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Cluster Analysis */}
        <div className="rounded-lg p-4" style={{ background: cardBg, border: `1px solid ${borderC}` }}>
          <div className="flex items-center gap-2 mb-4">
            <Eye className="h-4 w-4" style={{ color: '#a78bfa' }} />
            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: labelColor }}>Cluster Analysis</span>
          </div>
          {clusters.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-6">
              <Ship className="h-6 w-6 mb-2" style={{ color: darkMode ? '#1e3a5f' : '#cbd5e1' }} />
              <p className="text-xs" style={{ color: darkMode ? '#334155' : '#94a3b8' }}>
                {detections.length === 0 ? 'No detection data' : 'No ship clusters detected'}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {clusters.map((cl, i) => (
                <div key={i} className="p-3 rounded flex items-center gap-3"
                  style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)' }}>
                  <div className="w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold"
                    style={{ background: 'rgba(167,139,250,0.12)', color: '#a78bfa' }}>{cl.count}</div>
                  <div>
                    <div className="text-xs font-medium" style={{ color: '#a78bfa' }}>Fleet Cluster #{i + 1}</div>
                    <div className="text-xs font-mono mt-0.5" style={{ color: labelColor }}>
                      {cl.center.lat.toFixed(4)}°N, {cl.center.lon.toFixed(4)}°E
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      </div>
    </div>
  );
};
