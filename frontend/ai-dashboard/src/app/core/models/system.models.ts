export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'shutting_down' | 'unreachable';
  timestamp: number;
  app_env: string;
  demo_mode?: boolean;
}

export interface SystemReadyState {
  status: 'ready' | 'initializing' | 'unavailable';
  components: {
    central_orchestrator: boolean;
    device_gateway: boolean;
    input_gateway: boolean;
    fusion_engine: boolean;
    world_store: boolean;
    goal_manager: boolean;
    cognitive_runtime: boolean;
  };
  simulation_mode: boolean;
  demo_mode?: boolean;
}

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';
