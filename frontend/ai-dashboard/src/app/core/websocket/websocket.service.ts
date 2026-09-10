import { Injectable, OnDestroy, signal } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import { ApiClientService } from '../api/api-client.service';
import { CognitiveEvent, ConnectionStatus } from '../models';

@Injectable({
  providedIn: 'root'
})
export class WebSocketService implements OnDestroy {
  private socket: WebSocket | null = null;
  private pingIntervalId: any = null;
  private reconnectTimeoutId: any = null;
  private reconnectAttempts = 0;
  private maxReconnectDelayMs = 10000;
  private isDestroyed = false;

  public connectionStatus = signal<ConnectionStatus>('disconnected');
  private messageSubject = new Subject<CognitiveEvent>();
  public messages$: Observable<CognitiveEvent> = this.messageSubject.asObservable();

  constructor(private apiClient: ApiClientService) {}

  connect(): void {
    if (typeof window === 'undefined' || typeof WebSocket === 'undefined') {
      return; // SSR protection
    }

    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.connectionStatus.set('connecting');

    const baseUrl = this.apiClient.getBaseUrl();
    const wsProto = baseUrl.startsWith('https') ? 'wss' : 'ws';
    const host = baseUrl.replace(/^https?:\/\//, '');
    const token = encodeURIComponent(this.apiClient.getAuthToken());
    const wsUrl = `${wsProto}://${host}/api/v1/telemetry?token=${token}`;

    try {
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        if (this.isDestroyed) {
          this.disconnect();
          return;
        }
        this.reconnectAttempts = 0;
        this.connectionStatus.set('connected');
        this.startKeepAlive();
      };

      this.socket.onmessage = (event) => {
        try {
          if (event.data === 'pong') {
            return;
          }
          const parsed = JSON.parse(event.data);
          if (parsed && typeof parsed === 'object') {
            this.messageSubject.next(parsed as CognitiveEvent);
          }
        } catch {
          // Discard malformed frames safely
        }
      };

      this.socket.onerror = () => {
        this.connectionStatus.set('error');
      };

      this.socket.onclose = () => {
        this.stopKeepAlive();
        if (!this.isDestroyed) {
          this.connectionStatus.set('disconnected');
          this.scheduleReconnect();
        }
      };
    } catch {
      this.connectionStatus.set('error');
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    this.stopKeepAlive();
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }
    if (this.socket) {
      this.socket.onopen = null;
      this.socket.onmessage = null;
      this.socket.onerror = null;
      this.socket.onclose = null;
      try {
        this.socket.close();
      } catch {
        // Ignore close exceptions
      }
      this.socket = null;
    }
    this.connectionStatus.set('disconnected');
  }

  private startKeepAlive(): void {
    this.stopKeepAlive();
    this.pingIntervalId = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try {
          this.socket.send('ping');
        } catch {
          // Socket write failed
        }
      }
    }, 15000);
  }

  private stopKeepAlive(): void {
    if (this.pingIntervalId) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.isDestroyed) return;
    if (this.reconnectTimeoutId) return;

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), this.maxReconnectDelayMs);

    this.reconnectTimeoutId = setTimeout(() => {
      this.reconnectTimeoutId = null;
      if (!this.isDestroyed) {
        this.connect();
      }
    }, delay);
  }

  ngOnDestroy(): void {
    this.isDestroyed = true;
    this.disconnect();
    this.messageSubject.complete();
  }
}
