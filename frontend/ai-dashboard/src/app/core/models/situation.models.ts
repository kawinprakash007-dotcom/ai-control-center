import { GeoLocation, ProductType } from './product.models';

export type SituationSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFORMATIONAL';
export type SituationCategory = 'SECURITY' | 'SAFETY' | 'OPERATIONAL' | 'ENVIRONMENTAL' | 'SYSTEM';
export type SituationStatus = 'DETECTED' | 'VERIFYING' | 'CONFIRMED' | 'RESOLVED' | 'DISMISSED' | 'ACTIVE' | 'INVESTIGATING' | 'MITIGATED';

export interface SituationEvidence {
  evidence_id: string;
  observation_id?: string;
  source_id?: string;
  source_device_id?: string;
  modality: string;
  evidence_weight?: number;
  confidence?: number;
  timestamp: number;
  concise_summary?: string;
  description?: string;
}

export interface Situation {
  situation_id: string;
  category?: SituationCategory;
  severity: SituationSeverity;
  title: string;
  description: string;
  status: SituationStatus;
  confidence: number;
  primary_source_id?: string;
  participating_sources?: string[];
  participating_products?: string[];
  involved_products?: ProductType[];
  evidence?: SituationEvidence[];
  location?: GeoLocation;
  created_at?: number;
  detected_at?: number;
  updated_at?: number;
  has_contradiction?: boolean;
  is_stale?: boolean;
  resolved_at?: number;
  correlation_id?: string;
  recommended_action?: string;
}

export interface MultiProductSituation {
  multi_situation_id: string;
  title: string;
  category: SituationCategory;
  severity: SituationSeverity;
  confidence: number;
  participating_products: ProductType[];
  child_situations: Situation[];
  corroboration_score: number;
  created_at: number;
}
