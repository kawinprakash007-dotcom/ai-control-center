import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ApiClientService } from './api-client.service';
import {
  SystemHealth,
  SystemReadyState,
  ChatResponse,
  DeviceCommandRequest,
  DeviceCommandResult,
  WorldState,
  Trace,
} from '../models';

@Injectable({
  providedIn: 'root'
})
export class AtlasApiService {
  constructor(private apiClient: ApiClientService) {}

  getHealth(): Observable<SystemHealth> {
    return this.apiClient.get<SystemHealth>('/api/v1/health', false);
  }

  getReady(): Observable<SystemReadyState> {
    return this.apiClient.get<SystemReadyState>('/api/v1/ready', false);
  }

  postChat(message: string, sessionId: string = 'dashboard_session'): Observable<ChatResponse> {
    return this.apiClient.post<ChatResponse>(
      '/api/v1/chat',
      {
        message,
        session_id: sessionId,
        user_id: 'operator'
      },
      true,
      this.apiClient.getChatTimeoutMs()
    );
  }

  sendChat(payload: { message: string; session_id?: string }): Observable<ChatResponse> {
    return this.postChat(payload.message, payload.session_id || 'dashboard_session');
  }

  getDevices(): Observable<{ devices: any[] }> {
    return this.apiClient.get<{ devices: any[] }>('/api/v1/devices');
  }

  getDevice(deviceId: string): Observable<{ device: any; status: any }> {
    return this.apiClient.get<{ device: any; status: any }>(`/api/v1/devices/${encodeURIComponent(deviceId)}`);
  }

  dispatchDeviceCommand(
    deviceId: string,
    command: DeviceCommandRequest
  ): Observable<DeviceCommandResult> {
    return this.apiClient.post<DeviceCommandResult>(
      `/api/v1/devices/${encodeURIComponent(deviceId)}/command`,
      command
    );
  }

  sendCommand(
    deviceId: string,
    command: DeviceCommandRequest
  ): Observable<DeviceCommandResult> {
    return this.dispatchDeviceCommand(deviceId, command);
  }

  getWorldState(): Observable<WorldState> {
    return this.apiClient.get<WorldState>('/api/v1/world/state');
  }

  getGoals(): Observable<{ goals: any[] }> {
    return this.apiClient.get<{ goals: any[] }>('/api/v1/goals');
  }

  getTraces(limit: number = 50): Observable<{ traces: Trace[] }> {
    return this.apiClient.get<{ traces: Trace[] }>(`/api/v1/traces?limit=${limit}`);
  }

  postObservation(observation: any): Observable<any> {
    return this.apiClient.post<any>('/api/v1/ingress/observation', observation);
  }
}
