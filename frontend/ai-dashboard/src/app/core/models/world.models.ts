import { GeoLocation } from './product.models';

export interface WorldEntity {
  entity_id: string;
  entity_type: string;
  name: string;
  status: string;
  location?: GeoLocation;
  properties: Record<string, any>;
  confidence: number;
  last_updated: number;
}

export interface WorldCondition {
  condition_id: string;
  entity_id: string;
  property_name: string;
  value: any;
  confidence: number;
  source_id: string;
  timestamp: number;
  is_active: boolean;
}

export interface WorldRelationship {
  relationship_id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  properties: Record<string, any>;
  confidence: number;
}

export interface WorldState {
  state_id: string;
  version: number;
  timestamp: number;
  entities: WorldEntity[];
  conditions: WorldCondition[];
  relationships: WorldRelationship[];
  metadata: Record<string, any>;
}
