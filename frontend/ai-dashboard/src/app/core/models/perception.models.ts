import { GeoLocation } from './product.models';

export type ModalityType =
  | 'IMAGE'
  | 'AUDIO'
  | 'TEXT'
  | 'TELEMETRY'
  | 'GPS'
  | 'EVENT'
  | 'RADAR'
  | 'LIDAR';

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  confidence?: number;
  label?: string;
}

export interface PerceptionEvidence {
  evidence_id: string;
  source_id: string;
  modality: ModalityType;
  timestamp: number;
  confidence: number;
  label?: string;
  semantic_type?: string;
  attributes?: Record<string, any>;
  bounding_box?: BoundingBox;
  location?: GeoLocation;
  is_stale?: boolean;
  correlation_id?: string;
}

export type EvidenceRelationType =
  | 'CORROBORATES'
  | 'SUPPORTS'
  | 'CONTRADICTS'
  | 'DUPLICATES'
  | 'PRECEDES'
  | 'SAME_ENTITY_CANDIDATE';

export interface EvidenceRelationship {
  relation_type: EvidenceRelationType;
  source_evidence_id: string;
  target_evidence_id: string;
  confidence: number;
  reason?: string;
  temporal_relation?: string;
}

export interface FusionCluster {
  cluster_id: string;
  evidence_ids: string[];
  source_ids: string[];
  modalities: ModalityType[];
  confidence: number;
  contradiction_count: number;
  source_diversity: number;
}

export interface PerceptionResult {
  request_id: string;
  input_id: string;
  status: string;
  evidence: PerceptionEvidence[];
  created_at: number;
}
