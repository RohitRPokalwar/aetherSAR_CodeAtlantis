import React from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { ProcessingStage } from './types';

interface ProcessingStatusBarProps {
  stage: ProcessingStage;
  progress: number;
  tileCount: number;
  totalTiles: number;
  detectionCount: number;
}

const STAGES = [
  { label: 'Preprocessing', detail: 'Radiometric calibration & speckle filtering' },
  { label: 'Tiled Inference', detail: 'YOLO-SAR v10 inference on image tiles' },
  { label: 'Post-processing', detail: 'NMS merging & coordinate projection' },
];

export const ProcessingStatusBar: React.FC<ProcessingStatusBarProps> = ({
  stage, progress, tileCount, totalTiles, detectionCount,
}) => {
  if (stage === 0) return null;

  const currentStageIdx = stage - 1; // 0-indexed
  const displayTotalTiles = totalTiles || 64;

  return (
    <div className="flex-none border-b"
      style={{ background: '#060d1a', borderColor: '#1a2d45' }}>

      {/* Top progress strip */}
      <div className="h-0.5 relative" style={{ background: '#0f1d30' }}>
        <div
          className="h-full transition-all duration-300"
          style={{
            width: stage === 4 ? '100%' : `${((currentStageIdx) / 3 + progress / 300) * 100}%`,
            background: stage === 4
              ? 'linear-gradient(90deg, #22c55e, #00c8ff)'
              : 'linear-gradient(90deg, #0066cc, #00c8ff)',
            boxShadow: '0 0 8px rgba(0,200,255,0.6)',
          }} />
      </div>

      <div className="flex items-center gap-6 px-4 py-2">
        {/* Stage label */}
        <div className="flex items-center gap-2 min-w-0">
          {stage < 4 ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-none" style={{ color: '#00c8ff' }} />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 flex-none" style={{ color: '#22c55e' }} />
          )}
          <div>
            <span className="text-xs font-medium" style={{ color: stage === 4 ? '#22c55e' : '#00c8ff' }}>
              {stage === 4 ? 'COMPLETE' : `STAGE ${stage}/3`}
            </span>
            <span className="text-xs ml-2" style={{ color: '#475569' }}>
              {stage === 4 ? 'Post-processing' : STAGES[Math.min(currentStageIdx, 2)].label}
            </span>
            {stage === 2 && (
              <span className="text-xs ml-2" style={{ color: '#334155' }}>
                {tileCount}/{displayTotalTiles} tiles
              </span>
            )}
          </div>
        </div>

        <div className="w-px h-4 flex-none" style={{ background: '#1a2d45' }} />

        {/* Stage pipeline — only 3 stages */}
        <div className="flex items-center gap-3 flex-1 overflow-x-auto">
          {STAGES.map((s, i) => {
            const stageNum = i + 1;
            const isDone = stage > stageNum || stage === 4;
            const isActive = stage === stageNum;

            return (
              <React.Fragment key={s.label}>
                <div className="flex items-center gap-1.5 whitespace-nowrap">
                  {isDone ? (
                    <CheckCircle2 className="h-3 w-3 flex-none" style={{ color: '#22c55e' }} />
                  ) : isActive ? (
                    <Loader2 className="h-3 w-3 animate-spin flex-none" style={{ color: '#00c8ff' }} />
                  ) : (
                    <Circle className="h-3 w-3 flex-none" style={{ color: '#1e3a5f' }} />
                  )}
                  <span className="text-xs"
                    style={{ color: isDone ? '#22c55e' : isActive ? '#00c8ff' : '#334155' }}>
                    {s.label}
                  </span>
                </div>
                {i < STAGES.length - 1 && (
                  <div className="w-6 h-px flex-none"
                    style={{ background: isDone ? 'rgba(34,197,94,0.4)' : '#1a2d45' }} />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Progress percent */}
        <div className="flex-none">
          {stage < 4 && (
            <span className="text-xs font-mono" style={{ color: '#334155' }}>
              {Math.round(((currentStageIdx / 3) + (progress / 300)) * 100)}%
            </span>
          )}
          {stage === 4 && (
            <span className="text-xs" style={{ color: '#22c55e' }}>
              {detectionCount} ships detected
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
