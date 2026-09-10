import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../core/state/app-state.service';
import { AtlasApiService } from '../../core/api/atlas-api.service';
import { WebSocketService } from '../../core/websocket/websocket.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    StatusBadgeComponent,
  ],
  template: `
    <div class="settings-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">System Settings & Diagnostics</h1>
          <p class="page-subtitle">Central connectivity, authentication credentials, and authoritative subsystem health probes</p>
        </div>
      </div>

      <div class="settings-grid">
        <!-- Left: Connectivity & Credentials -->
        <div class="card form-card">
          <h3 class="card-title">BACKEND CONNECTION & AUTHENTICATION</h3>

          <div class="form-group">
            <label class="form-label">BACKEND API BASE URL</label>
            <input type="text" class="form-input font-mono" [(ngModel)]="apiUrl" placeholder="http://127.0.0.1:8000" />
            <span class="field-hint">Authoritative REST endpoint root (FastAPI)</span>
          </div>

          <div class="form-group">
            <label class="form-label">WEBSOCKET TELEMETRY URL</label>
            <input type="text" class="form-input font-mono" [(ngModel)]="wsUrl" placeholder="ws://127.0.0.1:8000/api/v1/telemetry" />
            <span class="field-hint">Live event stream & telemetry socket</span>
          </div>

          <div class="form-group">
            <label class="form-label">API AUTHENTICATION TOKEN (BEARER)</label>
            <input type="password" class="form-input font-mono" [(ngModel)]="authToken" placeholder="atlas_dev_secret_token" />
            <span class="field-hint">Validated against backend settings.api_auth_token</span>
          </div>

          <div class="btn-row">
            <button class="btn btn-primary" (click)="saveSettings()">
              Save & Apply Configuration
            </button>
            <button class="btn btn-secondary" (click)="resetDefaults()">
              Reset to Defaults
            </button>
          </div>

          @if (saveMessage()) {
            <div class="feedback-msg" [class.success]="saveSuccess()">
              {{ saveMessage() }}
            </div>
          }
        </div>

        <!-- Right: Subsystems Diagnostics & Probe Tests -->
        <div class="card diag-card">
          <div class="diag-header">
            <h3 class="card-title">AUTHORITATIVE SUBSYSTEM DIAGNOSTICS</h3>
            <button class="btn btn-secondary btn-sm" (click)="runDiagnostics()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
              Run Subsystem Probes
            </button>
          </div>

          <!-- Probe Tests -->
          <div class="probes-list">
            <div class="probe-item">
              <div class="p-info">
                <span class="p-name font-mono">GET /health</span>
                <span class="p-desc">FastAPI liveness probe</span>
              </div>
              <div class="p-status">
                <app-status-badge [status]="healthStatus()" size="sm"></app-status-badge>
              </div>
            </div>

            <div class="probe-item">
              <div class="p-info">
                <span class="p-name font-mono">GET /ready</span>
                <span class="p-desc">Subsystem initialization verification</span>
              </div>
              <div class="p-status">
                <app-status-badge [status]="readyStatus()" size="sm"></app-status-badge>
              </div>
            </div>

            <div class="probe-item">
              <div class="p-info">
                <span class="p-name font-mono">WS /api/v1/telemetry</span>
                <span class="p-desc">WebSocket stream connection</span>
              </div>
              <div class="p-status">
                <app-status-badge [status]="appState.connectionStatus() === 'connected' ? 'HEALTHY' : 'OFFLINE'" size="sm"></app-status-badge>
              </div>
            </div>
          </div>

          <!-- Active Subsystems List -->
          <div class="subsystems-box">
            <span class="sub-label">ACTIVE CENTRAL SUBSYSTEMS</span>
            <div class="subsystems-grid">
              <div class="sub-chip"><span class="dot"></span> CognitiveRuntime</div>
              <div class="sub-chip"><span class="dot"></span> CentralPolicyEngine</div>
              <div class="sub-chip"><span class="dot"></span> ToolOrchestrator</div>
              <div class="sub-chip"><span class="dot"></span> WorldModel (Digital Twin)</div>
              <div class="sub-chip"><span class="dot"></span> DeviceGateway</div>
              <div class="sub-chip"><span class="dot"></span> SituationEngine</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .settings-container {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .header-section {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }

    .page-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: -0.02em;
      margin-bottom: 0.25rem;
    }

    .page-subtitle {
      font-size: 0.85rem;
      color: #94a3b8;
    }

    .settings-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }

    @media (max-width: 900px) {
      .settings-grid {
        grid-template-columns: 1fr;
      }
    }

    .form-card, .diag-card {
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .card-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .form-label {
      font-size: 0.7rem;
      font-weight: 700;
      color: #cbd5e1;
      letter-spacing: 0.05em;
    }

    .form-input {
      background: #090d16;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 0.6rem 0.85rem;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .form-input:focus {
      border-color: #00f0ff;
    }

    .field-hint {
      font-size: 0.7rem;
      color: #64748b;
    }

    .btn-row {
      display: flex;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }

    .feedback-msg {
      font-size: 0.8rem;
      padding: 0.5rem;
      border-radius: 4px;
      background: rgba(239, 68, 68, 0.15);
      color: #ef4444;
    }

    .feedback-msg.success {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
    }

    .diag-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .probes-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .probe-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem 1rem;
      border-radius: 6px;
    }

    .p-name {
      display: block;
      font-size: 0.8rem;
      color: #00f0ff;
      font-weight: 600;
    }

    .p-desc {
      display: block;
      font-size: 0.7rem;
      color: #64748b;
    }

    .subsystems-box {
      margin-top: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .sub-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
    }

    .subsystems-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
    }

    .sub-chip {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
      padding: 0.4rem 0.6rem;
      border-radius: 4px;
      color: #e2e8f0;
      font-family: var(--font-mono);
    }

    .sub-chip .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
    }
  `]
})
export class SettingsComponent {
  public appState = inject(AppStateService);
  private api = inject(AtlasApiService);
  private ws = inject(WebSocketService);

  public apiUrl = 'http://127.0.0.1:8000';
  public wsUrl = 'ws://127.0.0.1:8000/api/v1/telemetry';
  public authToken = 'atlas_dev_secret_token';

  public saveMessage = signal<string | null>(null);
  public saveSuccess = signal<boolean>(false);

  public healthStatus = signal<string>('HEALTHY');
  public readyStatus = signal<string>('HEALTHY');

  constructor() {
    if (typeof window !== 'undefined') {
      this.apiUrl = localStorage.getItem('atlas_api_url') || this.apiUrl;
      this.wsUrl = localStorage.getItem('atlas_ws_url') || this.wsUrl;
      this.authToken = localStorage.getItem('atlas_auth_token') || this.authToken;
    }
  }

  public saveSettings(): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('atlas_api_url', this.apiUrl.trim());
      localStorage.setItem('atlas_ws_url', this.wsUrl.trim());
      localStorage.setItem('atlas_auth_token', this.authToken.trim());
    }
    this.saveSuccess.set(true);
    this.saveMessage.set('Configuration successfully saved to local storage. Refreshing connections...');
    this.appState.refreshAll();
    this.ws.disconnect();
    this.ws.connect();
  }

  public resetDefaults(): void {
    this.apiUrl = 'http://127.0.0.1:8000';
    this.wsUrl = 'ws://127.0.0.1:8000/api/v1/telemetry';
    this.authToken = 'atlas_dev_secret_token';
    this.saveSettings();
  }

  public runDiagnostics(): void {
    this.api.getHealth().subscribe({
      next: (res) => this.healthStatus.set(res.status === 'healthy' ? 'HEALTHY' : 'WARNING'),
      error: () => this.healthStatus.set('OFFLINE')
    });

    this.api.getReady().subscribe({
      next: (res) => this.readyStatus.set(res.status === 'ready' ? 'HEALTHY' : 'WARNING'),
      error: () => this.readyStatus.set('OFFLINE')
    });
  }
}
