export type ProductType = 'VISION' | 'GLASS' | 'DRONE' | 'ROVER';

export type ProductRole =
  | 'FIXED_STATIONARY'
  | 'MOBILE_RECON'
  | 'AERIAL_SURVEILLANCE'
  | 'GROUND_PATROL'
  | 'HYBRID'
  | 'OBSERVATION_SOURCE';

export type ProductHealth = 'HEALTHY' | 'DEGRADED' | 'FAULT' | 'UNKNOWN';
export type ProductConnectivity = 'ONLINE' | 'DISCONNECTED' | 'DEGRADED' | 'UNKNOWN';

export interface GeoLocation {
  latitude: number;
  longitude: number;
  altitude?: number;
}

export interface ProductCapability {
  capability_name: string;
  description?: string;
  parameters_schema?: Record<string, any>;
  schema_version?: string;
  is_destructive?: boolean;
}

export interface ProductTelemetry {
  timestamp: number;
  battery_level?: number;
  metrics?: Record<string, any>;
  health_status?: ProductHealth;
  connectivity?: ProductConnectivity;
  temperature_celsius?: number;
  location?: GeoLocation;
  speed_mps?: number;
  heading_degrees?: number;
  is_stale?: boolean;
}

export interface Product {
  device_id: string;
  display_name: string;
  device_type: string;
  product_type: ProductType;
  product_role: ProductRole;
  contract_version: string;
  is_simulation: boolean;
  connectivity_status: ProductConnectivity;
  health_status: ProductHealth;
  capabilities: ProductCapability[];
  home_location?: GeoLocation;
  current_location?: GeoLocation;
  telemetry?: ProductTelemetry;
  last_heartbeat?: number;
  registered_at: number;
  metadata?: Record<string, any>;
}
