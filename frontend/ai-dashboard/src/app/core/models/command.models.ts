export interface ChatRequest {
  message: string;
  session_id?: string;
  user_id?: string;
}

export interface ChatResponse {
  turn_id: string;
  status: string;
  response: string;
  execution_time: number;
  trace_id: string;
}

export interface DeviceCommandRequest {
  capability?: string;
  command?: string;
  action?: string;
  parameters?: Record<string, any>;
  priority?: string;
  correlation_id?: string;
}

export interface PolicyEvaluation {
  allowed: boolean;
  reason?: string;
  requires_confirmation?: boolean;
}

export interface DeviceCommandResult {
  command_id?: string;
  device_id?: string;
  success?: boolean;
  status?: string;
  message?: string;
  output?: any;
  result?: any;
  data?: any;
  error?: string;
  execution_time_ms?: number;
  policy_evaluation?: PolicyEvaluation;
}
