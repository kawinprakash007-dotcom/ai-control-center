export interface TraceEvent {
  event_id: string;
  stage: string;
  timestamp: number;
  duration_seconds?: number;
  status?: string;
  tool_name?: string;
  capability?: string;
  action?: string;
  details?: Record<string, any>;
}

export interface Trace {
  trace_id: string;
  turn_id?: string;
  id?: string;
  query?: string;
  user_message?: string;
  response?: string;
  duration_seconds?: number;
  execution_time_ms?: number;
  status: string;
  created_at?: number;
  timestamp?: number;
  events?: TraceEvent[];
  stages?: string[];
  steps?: any[];
  tools_called?: any[];
  metadata?: Record<string, any>;
}
