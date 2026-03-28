export interface Detection {
  id: string;
  lat: number | null;
  lon: number | null;
  confidence: number;
  type: 'Cargo' | 'Tanker' | 'Container' | 'Fishing' | 'Warship' | 'Unknown';
  lengthM: number;
  beamM: number;
  headingDeg: number;
  ais: string | null;
  rcs: number;
  bbox?: number[];
  pixelCenter?: number[];
}

export interface Viewport {
  centerLat: number;
  centerLon: number;
  scale: number; // pixels per degree
}

export interface ROIRegion {
  startLat: number;
  startLon: number;
  endLat: number;
  endLon: number;
}

export type ActiveTool = 'pan' | 'roi' | 'measure';

export type ProcessingStage = 0 | 1 | 2 | 3 | 4;
// 0 = idle, 1 = preprocessing, 2 = inference, 3 = postprocessing, 4 = complete

export interface PipelineConfig {
  modelArch: string;
  backbone: string;
  inputResolution: string;
  guardCells: number;
  trainingCells: number;
  pfaExp: number;
  confidenceThreshold: number;
  nmsIouThreshold: number;
  tileSize: number;
  tileOverlap: number;
  minShipLength: number;
  maxShipLength: number;
  mergeStrategy: string;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  modelArch: 'YOLO-SAR v10',
  backbone: 'CSP-DarkNet53',
  inputResolution: '1024×1024',
  guardCells: 8,
  trainingCells: 32,
  pfaExp: 6,
  confidenceThreshold: 65,
  nmsIouThreshold: 45,
  tileSize: 640,
  tileOverlap: 15,
  minShipLength: 20,
  maxShipLength: 600,
  mergeStrategy: 'WBF',
};

// API response types
export interface UploadResponse {
  fileId: string;
  filename: string;
  path: string;
  size: number;
  sizeHuman: string;
  width: number;
  height: number;
  bands: number;
  crs: string;
  dtype: string;
  isTiff: boolean;
}

export interface ProgressEvent {
  stage: ProcessingStage;
  progress: number;
  tileCount: number;
  totalTiles: number;
  detectionCount: number;
  done: boolean;
  error: string | null;
}

export interface DetectionSummary {
  totalDetections: number;
  tilesProcessed: number;
  processingTimeMs: number;
  imageWidth: number;
  imageHeight: number;
  coverageKm2: number;
}

export interface DetectionResponse {
  detections: Detection[];
  summary: DetectionSummary;
}

export interface FileMetadata {
  fileId: string;
  filename: string;
  width: number;
  height: number;
  crs: string;
  sizeHuman: string;
}
