import { ProductType } from './product.models';

export type MissionStatus =
  | 'PLANNED'
  | 'DISPATCHED'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'FAILED'
  | 'ABORTED'
  | 'CANCELLED';

export type MissionObjectiveType =
  | 'VERIFY_INCIDENT'
  | 'TRACK_TARGET'
  | 'PATROL_AREA'
  | 'INTERCEPT'
  | 'ASSIST_OPERATOR'
  | 'INSPECT_LOCATION';

export type ObjectiveStatus =
  | 'PENDING'
  | 'READY'
  | 'RUNNING'
  | 'PARTIAL'
  | 'COMPLETED'
  | 'FAILED'
  | 'BLOCKED'
  | 'SKIPPED'
  | 'CANCELLED';

export interface MissionObjective {
  objective_id: string;
  mission_id: string;
  type: MissionObjectiveType;
  description: string;
  assigned_product_id?: string;
  assigned_product_type?: ProductType;
  status: ObjectiveStatus;
  progress: number;
  result_summary?: string;
  blocker_reason?: string;
}

export interface Mission {
  mission_id: string;
  mission_type: string;
  description: string;
  current_status: MissionStatus;
  priority: string;
  progress: number;
  objectives: MissionObjective[];
  involved_product_ids: string[];
  trigger_situation_ids?: string[];
  created_at: number;
  updated_at: number;
  completed_at?: number;
}
