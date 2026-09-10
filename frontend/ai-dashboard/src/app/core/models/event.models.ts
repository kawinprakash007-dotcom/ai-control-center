export interface CognitiveEvent {
  type: string;
  event_id: string;
  event_type: string;
  turn_id: string;
  timestamp: number;
  stage: string;
  metadata?: Record<string, any>;
}

export interface AtlasEvent {
  event_id: string;
  source_id: string;
  event_type: string;
  priority: 'LOW' | 'NORMAL' | 'HIGH' | 'CRITICAL';
  product_id?: string;
  message: string;
  timestamp: number;
  correlation_id?: string;
  causation_id?: string;
  metadata?: Record<string, any>;
}
