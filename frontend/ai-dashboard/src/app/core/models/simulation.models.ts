export interface ScenarioCatalogItem {
  scenario_id: string;
  name?: string;
  title?: string;
  description: string;
  products?: string[];
  involved_products?: string[];
  step_count?: number;
  tags?: string[];
  initial_time?: number;
  modalities_tested?: string[];
  deterministic_hash?: string;
  expected_outcome?: string;
}

export interface ScenarioResult {
  scenario_id: string;
  success: boolean;
  simulated_duration: number;
  deterministic_hash: string;
  completed_steps: number;
  total_steps_executed: number;
  summary: string;
  assertions_passed?: number;
  assertions_evaluated?: number;
}

export interface CausalTraceItem {
  stage: string;
  observation_id?: string;
  source_id?: string;
  modality?: string;
  situation_id?: string;
  category?: string;
  severity?: string;
  transition_id?: string;
  entity_id?: string;
  property_name?: string;
  transition_type?: string;
  event_id?: string;
  decision_type?: string;
  goal_id?: string;
  title?: string;
  turn_id?: string;
  capability?: string;
  action?: string;
  success?: boolean;
  call_id?: string;
  correlation_id?: string;
  causation_id?: string;
  [key: string]: any;
}

export interface SimulationSituation {
  situation_id: string;
  category: string;
  severity: string;
  status: string;
  title: string;
  description?: string;
}

export interface SimulationTransition {
  transition_id: string;
  entity_id: string;
  property_name: string;
  transition_type: string;
  from_version?: number;
  to_version?: number;
}

export interface SimulationGoal {
  goal_id: string;
  title: string;
  status: string;
  priority: string;
}

export interface SimulationToolResult {
  capability: string;
  action: string;
  success: boolean;
  call_id?: string;
}

export interface SimulationResultDetails {
  scenario_id: string;
  status: string;
  observation_id: string;
  correlation_id: string;
  causation_id?: string;
  cycle_id: string;
  observations_ingested_count: number;
  situations_fused_count: number;
  world_transitions_count: number;
  events_evaluated_count?: number;
  autonomy_decisions_count?: number;
  goals_created_count: number;
  tool_results_count: number;
  duration_seconds: number;
  timestamp: number;
  causal_trace?: CausalTraceItem[];
  situations?: SimulationSituation[];
  world_transitions?: SimulationTransition[];
  goals?: SimulationGoal[];
  tool_results?: SimulationToolResult[];
  before_world_version?: number;
  before_entities_count?: number;
  before_conditions_count?: number;
}
