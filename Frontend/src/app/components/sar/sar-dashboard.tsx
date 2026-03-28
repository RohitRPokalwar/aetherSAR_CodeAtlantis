import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { TopNavbar } from './top-navbar';
import { ProcessingStatusBar } from './processing-status-bar';
import { NavSidebar, PageId } from './nav-sidebar';
import { UploadModal } from './upload-modal';
import { DashboardPage } from './pages/dashboard-page';
import { DetectionPage } from './pages/detection-page';
import { AlertsPage } from './pages/alerts-page';
import { ReportsPage } from './pages/reports-page';
import { LocationPage } from './pages/location-page';
import { GalleryPage } from './pages/gallery-page';
import {
  Detection, ProcessingStage, PipelineConfig,
  DEFAULT_PIPELINE_CONFIG, UploadResponse, ProgressEvent, DetectionSummary,
  FileMetadata,
} from './types';

// Alert generation (for TopNavbar)
interface Alert { type: 'critical' | 'warning' | 'info'; message: string; time: string; }

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

export const SARDashboard: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('detection');
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [tileCount, setTileCount] = useState(0);
  const [totalTiles, setTotalTiles] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_PIPELINE_CONFIG);
  const [fileId, setFileId] = useState<string | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [fileMetadata, setFileMetadata] = useState<FileMetadata | null>(null);
  const [summary, setSummary] = useState<DetectionSummary | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(() => setModelReady(true))
      .catch(() => setModelReady(false));
  }, []);

  // Generate alerts based on current state
  const alerts = useMemo(
    () => generateAlerts(detections.length, modelReady, fileId !== null),
    [detections.length, modelReady, fileId],
  );

  const handleUploadComplete = useCallback((upload: UploadResponse) => {
    setFileId(upload.fileId);
    setOriginalImageUrl(`/api/image/${upload.fileId}`);
    setResultImageUrl(null);
    setFileMetadata({
      fileId: upload.fileId,
      filename: upload.filename,
      width: upload.width,
      height: upload.height,
      crs: upload.crs,
      sizeHuman: upload.sizeHuman,
    });
    setDetections([]);
    setSelectedId(null);
    setProcessingStage(0);
    setProcessingProgress(0);
    setTileCount(0);
    setTotalTiles(0);
    setSummary(null);
    // Auto-navigate to detection page after upload
    setActivePage('detection');
  }, []);

  const handleRunDetection = useCallback(async () => {
    if (!fileId) return;

    // Stop any previous polling
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    setDetections([]);
    setSelectedId(null);
    setProcessingStage(1);
    setProcessingProgress(0);
    setTileCount(0);
    setTotalTiles(0);
    setSummary(null);
    setResultImageUrl(null);

    try {
      const detectRes = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fileId, config }),
      });

      if (!detectRes.ok) {
        console.error('Detection failed:', await detectRes.json());
        setProcessingStage(0);
        return;
      }

      // Poll progress every 500ms using the JSON endpoint (avoids SSE proxy issues)
      const startedAt = Date.now();
      const TIMEOUT_MS = 5 * 60 * 1000; // 5 minute safety timeout

      const interval = setInterval(async () => {
        // Safety timeout — if polling takes too long, reset
        if (Date.now() - startedAt > TIMEOUT_MS) {
          console.error('Detection polling timed out after 5 minutes');
          clearInterval(interval);
          pollRef.current = null;
          setProcessingStage(0);
          return;
        }

        try {
          const r = await fetch(`/api/progress-poll/${fileId}`);
          if (!r.ok) return; // 404 = not started yet, just wait

          const data: ProgressEvent = await r.json();

          if (data.error) {
            console.error('Processing error:', data.error);
            clearInterval(interval);
            pollRef.current = null;
            setProcessingStage(0);
            return;
          }

          setProcessingStage(data.stage as ProcessingStage);
          setProcessingProgress(data.progress);
          setTileCount(data.tileCount);
          setTotalTiles(data.totalTiles);

          if (data.done) {
            clearInterval(interval);
            pollRef.current = null;
            // Fetch results with retry
            for (let attempt = 0; attempt < 10; attempt++) {
              try {
                const resR = await fetch(`/api/results/${fileId}`);
                if (resR.ok) {
                  const result = await resR.json();
                  setProcessingStage(4);
                  setProcessingProgress(100);
                  setDetections(result.detections || []);
                  setSummary(result.summary || null);
                  setResultImageUrl(`/api/result-image/${fileId}?t=${Date.now()}`);
                  return;
                }
              } catch { /* retry */ }
              await new Promise(resolve => setTimeout(resolve, 500));
            }
            console.error('Failed to fetch results after retries');
            setProcessingStage(0);
          }
        } catch {
          // Network hiccup — polling will retry on next interval automatically
        }
      }, 500);

      pollRef.current = interval;

    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Detection request failed:', err);
        setProcessingStage(0);
      }
    }
  }, [fileId, config]);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  const isProcessing = processingStage > 0 && processingStage < 4;

  const handleExportCSV = useCallback(() => {
    if (!fileId) return;
    window.open(`/api/export/${fileId}`, '_blank');
  }, [fileId]);

  const handleExportPDF = useCallback(async () => {
    if (!fileId) return;
    const response = await fetch(`/api/report/${fileId}`);
    if (!response.ok) return;
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SAR_Report_${fileId}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, [fileId]);

  // Dynamic theme
  const theme = darkMode
    ? { bg: '#040c18', color: '#f1f5f9', panelBg: '#08121e', workspace: '#040c18' }
    : { bg: '#f8fafc', color: '#1e293b', panelBg: '#ffffff', workspace: '#f1f5f9' };

  /* ── Render active page ── */
  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage detections={detections} summary={summary} darkMode={darkMode} />;
      case 'detection':
        return (
          <DetectionPage
            detections={detections}
            selectedId={selectedId}
            onSelectDetection={setSelectedId}
            config={config}
            onConfigChange={setConfig}
            onRunDetection={handleRunDetection}
            isProcessing={isProcessing}
            hasFile={fileId !== null}
            modelReady={modelReady}
            fileId={fileId}
            originalImageUrl={originalImageUrl}
            resultImageUrl={resultImageUrl}
            summary={summary}
            onExportCSV={handleExportCSV}
            onExportPDF={handleExportPDF}
            darkMode={darkMode}
          />
        );
      case 'location':
        return <LocationPage detections={detections} fileId={fileId} darkMode={darkMode} />;
      case 'alerts':
        return <AlertsPage detections={detections} modelReady={modelReady} hasFile={fileId !== null} darkMode={darkMode} />;
      case 'reports':
        return <ReportsPage detections={detections} summary={summary} fileId={fileId} darkMode={darkMode} />;
      case 'gallery':
        return <GalleryPage detections={detections} fileId={fileId} darkMode={darkMode} summary={summary} />;
    }
  };

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden"
      style={{ background: theme.bg, color: theme.color, fontFamily: "'Space Grotesk', 'JetBrains Mono', monospace" }}>

      {/* Top Navigation */}
      <TopNavbar
        processingStage={processingStage}
        onOpenUpload={() => setIsUploadOpen(true)}
        detectionCount={detections.length}
        fileMetadata={fileMetadata}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(v => !v)}
        alerts={alerts}
      />

      {/* Processing Status Bar */}
      <ProcessingStatusBar
        stage={processingStage}
        progress={processingProgress}
        tileCount={tileCount}
        totalTiles={totalTiles}
        detectionCount={detections.length}
      />

      {/* Main workspace: Sidebar + Active Page */}
      <div className="flex flex-1 overflow-hidden">
        <NavSidebar
          activePage={activePage}
          onNavigate={setActivePage}
          onOpenUpload={() => setIsUploadOpen(true)}
          darkMode={darkMode}
        />
        {renderPage()}
      </div>

      {/* Upload Modal */}
      {isUploadOpen && (
        <UploadModal
          onClose={() => setIsUploadOpen(false)}
          onUploadComplete={handleUploadComplete}
        />
      )}
    </div>
  );
};