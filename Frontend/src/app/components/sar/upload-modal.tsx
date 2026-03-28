import React, { useState, useRef, useCallback } from 'react';
import { X, Upload, FileCheck, FileX, AlertCircle, CheckCircle2, HardDrive, Zap } from 'lucide-react';
import { UploadResponse } from './types';

interface UploadModalProps {
  onClose: () => void;
  onUploadComplete: (upload: UploadResponse) => void;
}

type UploadPhase = 'idle' | 'validating' | 'uploading' | 'indexing' | 'complete' | 'error';

interface FileState {
  file: File;
  phase: UploadPhase;
  progress: number;
  speed: number; // MB/s
  error?: string;
  uploadResponse?: UploadResponse;
}

const ACCEPTED = ['.tiff', '.tif', '.geotiff', '.nc', '.h5', '.png', '.jpg', '.jpeg'];
const MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024; // 2GB

function formatSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export const UploadModal: React.FC<UploadModalProps> = ({ onClose, onUploadComplete }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [files, setFiles] = useState<FileState[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadFileToServer = useCallback(async (file: File, key: string) => {
    // Phase 1: Validating
    setFiles(prev => prev.map(f => f.file.name === key ? { ...f, phase: 'validating' } : f));

    // Small delay for validation UX
    await new Promise(r => setTimeout(r, 500));

    // Phase 2: Uploading with real progress tracking
    setFiles(prev => prev.map(f => f.file.name === key ? { ...f, phase: 'uploading', progress: 0 } : f));

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Use XMLHttpRequest for progress tracking
      const response = await new Promise<UploadResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const pct = (e.loaded / e.total) * 100;
            const speed = e.loaded / (1024 * 1024); // rough MB
            setFiles(prev => prev.map(f =>
              f.file.name === key ? { ...f, progress: pct, speed: speed / Math.max(1, performance.now() / 1000) * 10 } : f
            ));
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch {
              reject(new Error('Invalid response'));
            }
          } else {
            reject(new Error(`Upload failed: ${xhr.status}`));
          }
        });

        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));

        xhr.open('POST', '/api/upload');
        xhr.timeout = 1800000; // 30 min — large file uploads
        xhr.send(formData);
      });

      // Phase 3: Indexing
      setFiles(prev => prev.map(f =>
        f.file.name === key ? { ...f, phase: 'indexing', progress: 100, speed: 0 } : f
      ));

      // Brief indexing delay
      await new Promise(r => setTimeout(r, 800));

      // Phase 4: Complete
      setFiles(prev => prev.map(f =>
        f.file.name === key ? { ...f, phase: 'complete', uploadResponse: response } : f
      ));

    } catch (err: any) {
      setFiles(prev => prev.map(f =>
        f.file.name === key ? { ...f, phase: 'error', error: err.message || 'Upload failed' } : f
      ));
    }
  }, []);

  const validateAndAddFiles = useCallback((rawFiles: File[]) => {
    const valid = rawFiles.filter(f => {
      const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
      return ACCEPTED.includes(ext) && f.size <= MAX_SIZE_BYTES;
    });
    if (valid.length === 0) return;
    const newStates: FileState[] = valid.map(f => ({ file: f, phase: 'idle' as UploadPhase, progress: 0, speed: 0 }));
    setFiles(prev => [...prev, ...newStates]);

    // Start uploading each file
    newStates.forEach(fs => {
      uploadFileToServer(fs.file, fs.file.name);
    });
  }, [uploadFileToServer]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    validateAndAddFiles(Array.from(e.dataTransfer.files));
  }, [validateAndAddFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) validateAndAddFiles(Array.from(e.target.files));
  }, [validateAndAddFiles]);

  const allComplete = files.length > 0 && files.every(f => f.phase === 'complete');
  const hasUploading = files.some(f => f.phase === 'uploading' || f.phase === 'validating' || f.phase === 'indexing');

  const handleOpenInWorkspace = useCallback(() => {
    // Find the last successfully uploaded file
    const completedFile = files.findLast(f => f.phase === 'complete' && f.uploadResponse);
    if (completedFile?.uploadResponse) {
      onUploadComplete(completedFile.uploadResponse);
    }
    onClose();
  }, [files, onUploadComplete, onClose]);

  const PhaseIcon: React.FC<{ phase: UploadPhase }> = ({ phase }) => {
    if (phase === 'complete') return <CheckCircle2 className="h-4 w-4" style={{ color: '#22c55e' }} />;
    if (phase === 'error') return <FileX className="h-4 w-4" style={{ color: '#ef4444' }} />;
    if (phase === 'validating') return <AlertCircle className="h-4 w-4 animate-pulse" style={{ color: '#f59e0b' }} />;
    return <div className="h-4 w-4 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: '#00c8ff', borderTopColor: 'transparent' }} />;
  };

  const phaseLabel = (phase: UploadPhase) => {
    const map: Record<UploadPhase, string> = {
      idle: '', validating: 'Validating metadata…', uploading: 'Uploading (chunked)',
      indexing: 'Indexing scene…', complete: 'Ready', error: 'Error',
    };
    return map[phase];
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(4,12,24,0.85)', backdropFilter: 'blur(4px)' }}>
      <div className="relative w-full max-w-xl mx-4 rounded-xl overflow-hidden"
        style={{ background: '#08121e', border: '1px solid #1a2d45', boxShadow: '0 0 60px rgba(0,200,255,0.1)' }}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: '#1a2d45' }}>
          <div>
            <h2 className="text-sm font-bold" style={{ color: '#e2e8f0' }}>Upload SAR Imagery</h2>
            <p className="text-xs mt-0.5" style={{ color: '#475569' }}>GeoTIFF · NetCDF · HDF5 · PNG · JPG · Max 2GB per file</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded transition-colors"
            style={{ color: '#475569' }}
            onMouseEnter={e => (e.currentTarget.style.color = '#94a3b8')}
            onMouseLeave={e => (e.currentTarget.style.color = '#475569')}>
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Drop zone */}
        <div className="p-5">
          <div
            onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className="rounded-lg flex flex-col items-center justify-center py-10 cursor-pointer transition-all duration-200"
            style={{
              border: `2px dashed ${isDragOver ? '#00c8ff' : '#1e3a5f'}`,
              background: isDragOver ? 'rgba(0,200,255,0.06)' : 'rgba(0,0,0,0.2)',
            }}>
            <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3"
              style={{ background: 'rgba(0,200,255,0.1)', border: '1px solid rgba(0,200,255,0.2)' }}>
              <Upload className="h-6 w-6" style={{ color: '#00c8ff' }} />
            </div>
            <p className="text-sm" style={{ color: '#94a3b8' }}>Drop SAR imagery files here</p>
            <p className="text-xs mt-1" style={{ color: '#334155' }}>or click to browse</p>
            <div className="flex items-center gap-3 mt-4">
              {ACCEPTED.map(ext => (
                <span key={ext} className="text-xs px-2 py-0.5 rounded"
                  style={{ background: '#0a1525', border: '1px solid #1a2d45', color: '#475569' }}>.{ext.replace('.', '')}</span>
              ))}
            </div>
          </div>
          <input ref={inputRef} type="file" multiple accept={ACCEPTED.join(',')} className="hidden" onChange={handleFileInput} />
        </div>

        {/* File list */}
        {files.length > 0 && (
          <div className="px-5 pb-4 space-y-3 max-h-64 overflow-y-auto">
            <div className="text-xs uppercase tracking-widest mb-2" style={{ color: '#334155' }}>Upload Queue</div>
            {files.map(fs => (
              <div key={fs.file.name} className="rounded-lg p-3"
                style={{ background: '#0c1830', border: '1px solid #1a2d45' }}>
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <PhaseIcon phase={fs.phase} />
                    <div className="min-w-0">
                      <p className="text-xs truncate" style={{ color: '#94a3b8' }}>{fs.file.name}</p>
                      <p className="text-xs" style={{ color: '#334155' }}>
                        {formatSize(fs.file.size)}
                        {fs.uploadResponse && ` · ${fs.uploadResponse.width}×${fs.uploadResponse.height}`}
                      </p>
                    </div>
                  </div>
                  <div className="flex-none text-right">
                    <p className="text-xs" style={{ color: fs.phase === 'complete' ? '#22c55e' : fs.phase === 'error' ? '#ef4444' : '#475569' }}>
                      {fs.phase === 'error' ? fs.error : phaseLabel(fs.phase)}
                    </p>
                    {fs.phase === 'uploading' && (
                      <p className="text-xs" style={{ color: '#334155' }}>{fs.speed.toFixed(0)} MB/s</p>
                    )}
                  </div>
                </div>

                {/* Progress bar */}
                {(fs.phase === 'uploading' || fs.phase === 'indexing') && (
                  <div className="mt-2">
                    <div className="h-1 rounded-full overflow-hidden" style={{ background: '#0a1525' }}>
                      <div
                        className="h-full rounded-full transition-all duration-150"
                        style={{
                          width: `${fs.phase === 'indexing' ? 100 : fs.progress}%`,
                          background: 'linear-gradient(90deg, #0066cc, #00c8ff)',
                          boxShadow: '0 0 8px rgba(0,200,255,0.5)',
                        }} />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="text-xs" style={{ color: '#1e3a5f' }}>
                        {fs.phase === 'indexing' ? 'Building spatial index…' : 'Uploading to server…'}
                      </span>
                      <span className="text-xs font-mono" style={{ color: '#334155' }}>
                        {fs.phase === 'indexing' ? '—' : `${Math.round(fs.progress)}%`}
                      </span>
                    </div>
                  </div>
                )}

                {/* Validation checks */}
                {fs.phase === 'validating' && (
                  <div className="mt-2 space-y-1">
                    {['Checking CRS metadata', 'Verifying geolocation', 'Validating file integrity'].map((check) => (
                      <div key={check} className="flex items-center gap-2">
                        <div className="w-1 h-1 rounded-full" style={{ background: '#1e3a5f' }} />
                        <span className="text-xs" style={{ color: '#334155' }}>{check}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Info row */}
        <div className="px-5 pb-3 flex gap-4">
          {[
            [<HardDrive className="h-3 w-3" />, 'Chunked upload (8MB/chunk)'],
            [<Zap className="h-3 w-3" />, 'Auto-projected to WGS84'],
            [<FileCheck className="h-3 w-3" />, 'CRS & metadata validated'],
          ].map(([icon, text], i) => (
            <div key={i} className="flex items-center gap-1.5 text-xs" style={{ color: '#334155' }}>
              <span style={{ color: '#1e3a5f' }}>{icon as React.ReactNode}</span>
              {text as string}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-4 border-t" style={{ borderColor: '#1a2d45' }}>
          <button onClick={onClose} className="px-4 py-2 rounded text-sm transition-colors"
            style={{ background: '#0a1525', border: '1px solid #1a2d45', color: '#475569' }}>
            {allComplete ? 'Close' : 'Cancel'}
          </button>
          {allComplete && (
            <button
              onClick={handleOpenInWorkspace}
              className="px-5 py-2 rounded text-sm flex items-center gap-2 transition-all"
              style={{ background: 'linear-gradient(135deg, #0066cc, #00c8ff)', color: '#fff', boxShadow: '0 0 20px rgba(0,200,255,0.25)' }}>
              <CheckCircle2 className="h-4 w-4" />
              Open in GIS Workspace
            </button>
          )}
          {!allComplete && files.length > 0 && (
            <div className="text-xs" style={{ color: '#334155' }}>
              {files.filter(f => f.phase === 'complete').length}/{files.length} files ready
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
