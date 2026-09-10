import { Component, OnInit, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../../core/state/app-state.service';
import { AtlasApiService } from '../../../core/api/atlas-api.service';
import { Product, DeviceCommandRequest, DeviceCommandResult } from '../../../core/models';
import { StatusBadgeComponent } from '../../../shared/components/status-badge/status-badge.component';
import { TelemetryChartComponent } from '../../../shared/components/telemetry-chart/telemetry-chart.component';
import { ConfirmationDialogComponent } from '../../../shared/components/confirmation-dialog/confirmation-dialog.component';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-product-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    StatusBadgeComponent,
    TelemetryChartComponent,
    ConfirmationDialogComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="product-detail-container fade-in">
      <!-- Top Navigation / Header -->
      <div class="top-nav">
        <a routerLink="/products" class="back-link">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="19" y1="12" x2="5" y2="12"></line>
            <polyline points="12 19 5 12 12 5"></polyline>
          </svg>
          Back to Fleet Overview
        </a>
        <div class="actions">
          <button class="btn btn-secondary btn-sm" (click)="appState.refreshDevices()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Refresh Telemetry
          </button>
        </div>
      </div>

      @if (!product()) {
        <app-empty-state
          icon="hardware"
          title="Product Not Found"
          [message]="'No registered product matches ID: ' + deviceId()"
          actionText="Return to Fleet"
          routerLink="/products">
        </app-empty-state>
      } @else {
        <!-- Product Header Bar -->
        <div class="product-banner card">
          <div class="banner-left">
            <div class="product-avatar" [ngClass]="product()!.product_type.toLowerCase()">
              {{ product()!.product_type.substring(0, 2) }}
            </div>
            <div>
              <div class="meta-row">
                <span class="device-id font-mono">{{ product()!.device_id }}</span>
                <span class="version-tag">Contract v{{ product()!.contract_version }}</span>
                @if (product()!.is_simulation) {
                  <span class="sim-pill">DIGITAL TWIN SIM</span>
                }
              </div>
              <h1 class="product-title">{{ product()!.display_name }}</h1>
              <p class="role-desc">Role: <strong>{{ product()!.product_role }}</strong> | Type: <strong>{{ product()!.device_type }}</strong></p>
            </div>
          </div>
          <div class="banner-right">
            <div class="status-stack">
              <app-status-badge [status]="product()!.connectivity_status" size="md"></app-status-badge>
              <app-status-badge [status]="product()!.health_status" size="sm"></app-status-badge>
            </div>
          </div>
        </div>

        <!-- Main Content Grid -->
        <div class="detail-grid">
          <!-- Left Column: Gauges & Metrics -->
          <div class="metrics-col">
            <!-- Telemetry Gauges Grid -->
            <div class="gauges-grid">
              <div class="gauge-card card">
                <span class="g-title">BATTERY RESERVE</span>
                <div class="g-value font-mono" [style.color]="getBatteryColor(product()!.telemetry?.battery_level)">
                  {{ product()!.telemetry?.battery_level != null ? product()!.telemetry?.battery_level + '%' : 'N/A' }}
                </div>
                <div class="progress-bar-bg">
                  <div class="progress-bar-fill" [style.width.%]="product()!.telemetry?.battery_level || 0"
                       [style.background]="getBatteryColor(product()!.telemetry?.battery_level)"></div>
                </div>
                <span class="g-sub">Nominal threshold: 20%</span>
              </div>

              <div class="gauge-card card">
                <span class="g-title">CORE TEMPERATURE</span>
                <div class="g-value font-mono">
                  {{ product()!.telemetry?.temperature_celsius ?? 24.5 }}°C
                </div>
                <div class="progress-bar-bg">
                  <div class="progress-bar-fill" [style.width.%]="((product()!.telemetry?.temperature_celsius ?? 25) / 80) * 100"
                       style="background: #00f0ff;"></div>
                </div>
                <span class="g-sub">Operating range: -10°C to 70°C</span>
              </div>

              <div class="gauge-card card">
                <span class="g-title">VELOCITY / SPEED</span>
                <div class="g-value font-mono">
                  {{ product()!.telemetry?.speed_mps ?? 0 }} <span class="unit">m/s</span>
                </div>
                <span class="g-sub">Heading: {{ product()!.telemetry?.heading_degrees ?? 0 }}° North</span>
              </div>

              <div class="gauge-card card">
                <span class="g-title">HEARTBEAT FRESHNESS</span>
                <div class="g-value font-mono" style="color: #10b981;">
                  {{ formatFreshness(product()!.last_heartbeat) }}
                </div>
                <span class="g-sub">Interval: 1.0s Authoritative</span>
              </div>
            </div>

            <!-- Spatial / Location Card -->
            <div class="card spatial-card">
              <h3 class="card-title">SPATIAL TELEMETRY & COORDINATES</h3>
              <div class="coords-box font-mono">
                <div class="coord-item">
                  <span class="c-axis">X (LAT / EAST):</span>
                  <span class="c-val">{{ product()!.telemetry?.location?.latitude ?? product()!.home_location?.latitude ?? 0.0 }}</span>
                </div>
                <div class="coord-item">
                  <span class="c-axis">Y (LON / NORTH):</span>
                  <span class="c-val">{{ product()!.telemetry?.location?.longitude ?? product()!.home_location?.longitude ?? 0.0 }}</span>
                </div>
                <div class="coord-item">
                  <span class="c-axis">Z (ALTITUDE):</span>
                  <span class="c-val">{{ product()!.telemetry?.location?.altitude ?? product()!.home_location?.altitude ?? 0.0 }} m</span>
                </div>
              </div>
            </div>

            <!-- Telemetry Chart -->
            <div class="card chart-card">
              <h3 class="card-title">TIME-SERIES TELEMETRY STREAM</h3>
              <app-telemetry-chart
                [data]="batteryHistory()"
                label="Battery Level (%)"
                strokeColor="#10b981"
                fillColor="rgba(16, 185, 129, 0.1)"
                [height]="110">
              </app-telemetry-chart>
            </div>
          </div>

          <!-- Right Column: Command Execution Panel -->
          <div class="commands-col">
            <div class="card command-panel">
              <div class="panel-header">
                <div>
                  <h3 class="card-title">AUTHORITATIVE DEVICE COMMAND</h3>
                  <p class="panel-sub">Dispatches via Central PolicyEngine & DeviceGateway</p>
                </div>
                <span class="policy-badge">POLICY GUARDED</span>
              </div>

              <!-- Capability Selection -->
              <div class="form-group">
                <label class="form-label">SELECT CAPABILITY / COMMAND</label>
                <select class="form-select" [(ngModel)]="selectedCapability">
                  @for (cap of product()!.capabilities; track cap.capability_name) {
                    <option [value]="cap.capability_name">{{ cap.capability_name }} — {{ cap.description }}</option>
                  }
                  @if (product()!.capabilities.length === 0) {
                    <option value="ping">ping — Standard health check ping</option>
                    <option value="status">status — Retrieve detailed hardware status</option>
                  }
                </select>
              </div>

              <!-- Parameters JSON Editor -->
              <div class="form-group">
                <label class="form-label">COMMAND PARAMETERS (JSON)</label>
                <textarea
                  class="form-textarea font-mono"
                  rows="4"
                  [(ngModel)]="commandParametersJson"
                  placeholder='{ "timeout_s": 10 }'></textarea>
                @if (jsonError()) {
                  <span class="error-hint">{{ jsonError() }}</span>
                }
              </div>

              <!-- Action Button -->
              <div class="action-row">
                <button
                  class="btn btn-primary w-full"
                  [disabled]="isExecuting()"
                  (click)="handlePreFlight()">
                  @if (isExecuting()) {
                    <span class="spinner-sm"></span> Dispatching...
                  } @else {
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Execute Command
                  }
                </button>
              </div>

              <!-- Execution Result Output -->
              @if (lastResult()) {
                <div class="result-box" [class.success]="lastResult()!.status === 'success'" [class.error]="lastResult()!.status === 'error'">
                  <div class="res-header">
                    <span class="res-status font-mono">STATUS: {{ lastResult()!.status | uppercase }}</span>
                    @if (lastResult()!.execution_time_ms) {
                      <span class="res-time font-mono">{{ lastResult()!.execution_time_ms }} ms</span>
                    }
                  </div>
                  @if (lastResult()!.policy_evaluation) {
                    <div class="policy-eval font-mono">
                      Policy Verdict: <span [style.color]="lastResult()!.policy_evaluation?.allowed ? '#10b981' : '#ef4444'">
                        {{ lastResult()!.policy_evaluation?.allowed ? 'APPROVED' : 'DENIED' }}
                      </span>
                      @if (lastResult()!.policy_evaluation?.reason) {
                        — {{ lastResult()!.policy_evaluation?.reason }}
                      }
                    </div>
                  }
                  <pre class="res-payload font-mono">{{ formatResultPayload(lastResult()!) }}</pre>
                </div>
              }
            </div>
          </div>
        </div>
      }

      <!-- Pre-flight Confirmation Dialog for High Risk Commands -->
      <app-confirmation-dialog
        [isOpen]="showConfirmDialog()"
        title="Authoritative Command Authorization"
        [message]="'Are you sure you want to execute high-impact command [' + selectedCapability + '] on device ' + (product()?.device_id || '') + '?'"
        confirmText="Authorize & Dispatch"
        cancelText="Abort"
        (confirmed)="executeCommand()"
        (cancelled)="showConfirmDialog.set(false)">
      </app-confirmation-dialog>
    </div>
  `,
  styles: [`
    .product-detail-container {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .top-nav {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .back-link {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      color: #94a3b8;
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      transition: color 0.15s;
    }

    .back-link:hover {
      color: #00f0ff;
    }

    .product-banner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.25rem;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.6) 100%);
      flex-wrap: wrap;
      gap: 1rem;
    }

    .banner-left {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .product-avatar {
      width: 52px;
      height: 52px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      font-weight: 800;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #fff;
    }

    .product-avatar.vision { background: rgba(0, 240, 255, 0.12); color: #00f0ff; border-color: rgba(0, 240, 255, 0.3); }
    .product-avatar.glass { background: rgba(168, 85, 247, 0.12); color: #a855f7; border-color: rgba(168, 85, 247, 0.3); }
    .product-avatar.drone { background: rgba(245, 158, 11, 0.12); color: #f59e0b; border-color: rgba(245, 158, 11, 0.3); }
    .product-avatar.rover { background: rgba(16, 185, 129, 0.12); color: #10b981; border-color: rgba(16, 185, 129, 0.3); }

    .meta-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.2rem;
    }

    .device-id {
      font-size: 0.8rem;
      color: #94a3b8;
    }

    .version-tag {
      font-size: 0.7rem;
      color: #64748b;
      font-family: var(--font-mono);
    }

    .sim-pill {
      font-size: 0.65rem;
      font-family: var(--font-mono);
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.3);
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      font-weight: 700;
    }

    .product-title {
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 0.2rem;
    }

    .role-desc {
      font-size: 0.8rem;
      color: #64748b;
    }

    .status-stack {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.4rem;
    }

    .detail-grid {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 1.25rem;
    }

    @media (max-width: 900px) {
      .detail-grid {
        grid-template-columns: 1fr;
      }
    }

    .metrics-col {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .gauges-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
    }

    .gauge-card {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .g-title {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
    }

    .g-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
    }

    .g-value .unit {
      font-size: 0.9rem;
      font-weight: 400;
      color: #94a3b8;
    }

    .progress-bar-bg {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 0.25rem;
    }

    .progress-bar-fill {
      height: 100%;
      transition: width 0.3s ease;
    }

    .g-sub {
      font-size: 0.7rem;
      color: #64748b;
      margin-top: 0.25rem;
    }

    .spatial-card {
      padding: 1rem;
    }

    .coords-box {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem;
      border-radius: 6px;
      margin-top: 0.5rem;
    }

    .coord-item {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    .c-axis {
      color: #64748b;
    }

    .c-val {
      color: #00f0ff;
      font-weight: 600;
    }

    .chart-card {
      padding: 1rem;
    }

    .card-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 0.75rem;
    }

    .commands-col {
      display: flex;
      flex-direction: column;
    }

    .command-panel {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }

    .panel-sub {
      font-size: 0.75rem;
      color: #64748b;
    }

    .policy-badge {
      font-size: 0.65rem;
      font-family: var(--font-mono);
      background: rgba(0, 240, 255, 0.1);
      color: #00f0ff;
      border: 1px solid rgba(0, 240, 255, 0.25);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 700;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .form-label {
      font-size: 0.7rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .form-select, .form-textarea {
      background: #090d16;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 0.6rem;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .form-select:focus, .form-textarea:focus {
      border-color: #00f0ff;
    }

    .form-textarea {
      resize: vertical;
      line-height: 1.4;
    }

    .error-hint {
      font-size: 0.7rem;
      color: #ef4444;
    }

    .result-box {
      background: rgba(0, 0, 0, 0.3);
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .result-box.success { border-color: rgba(16, 185, 129, 0.4); }
    .result-box.error { border-color: rgba(239, 68, 68, 0.4); }

    .res-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
    }

    .res-status {
      font-weight: 700;
      color: #10b981;
    }

    .result-box.error .res-status {
      color: #ef4444;
    }

    .res-time {
      color: #64748b;
    }

    .policy-eval {
      font-size: 0.75rem;
      color: #94a3b8;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 0.4rem;
    }

    .res-payload {
      font-size: 0.75rem;
      color: #cbd5e1;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 200px;
      overflow-y: auto;
    }

    .spinner-sm {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 0.4rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `]
})
export class ProductDetailComponent implements OnInit {
  private route = inject(ActivatedRoute);
  public appState = inject(AppStateService);
  private api = inject(AtlasApiService);

  public deviceId = signal<string>('');
  public selectedCapability = 'status';
  public commandParametersJson = '{\n  "mode": "standard"\n}';
  public jsonError = signal<string | null>(null);
  public isExecuting = signal<boolean>(false);
  public lastResult = signal<DeviceCommandResult | null>(null);
  public showConfirmDialog = signal<boolean>(false);

  public product = computed(() => {
    const id = this.deviceId();
    return this.appState.products().find(p => p.device_id === id) || null;
  });

  public batteryHistory = computed(() => {
    const id = this.deviceId();
    const history = this.appState.getTelemetryHistory(id);
    if (history.length === 0) return [70, 72, 75, 78, 80, 82, 85, 84];
    return history.map(t => t.battery_level ?? 80);
  });

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      const id = params.get('id') || '';
      this.deviceId.set(id);
      const prod = this.product();
      if (prod && prod.capabilities.length > 0) {
        this.selectedCapability = prod.capabilities[0].capability_name;
      }
    });
  }

  public handlePreFlight(): void {
    this.jsonError.set(null);
    let parsed: any = {};
    if (this.commandParametersJson.trim()) {
      try {
        parsed = JSON.parse(this.commandParametersJson);
      } catch (e: any) {
        this.jsonError.set('Invalid JSON parameters: ' + e.message);
        return;
      }
    }

    // Check if command is high risk (e.g. takeoff, emergency_stop, wipe, drive)
    const highRisk = ['takeoff', 'land', 'emergency_stop', 'drive', 'reboot'];
    if (highRisk.includes(this.selectedCapability.toLowerCase())) {
      this.showConfirmDialog.set(true);
    } else {
      this.executeCommand();
    }
  }

  public executeCommand(): void {
    this.showConfirmDialog.set(false);
    const prod = this.product();
    if (!prod) return;

    let params: any = {};
    try {
      params = JSON.parse(this.commandParametersJson);
    } catch {}

    const req: DeviceCommandRequest = {
      command: this.selectedCapability,
      parameters: params,
      priority: 'NORMAL',
    };

    this.isExecuting.set(true);
    this.api.sendCommand(prod.device_id, req).subscribe({
      next: (res) => {
        this.isExecuting.set(false);
        this.lastResult.set(res);
        this.appState.refreshDevices();
      },
      error: (err) => {
        this.isExecuting.set(false);
        this.lastResult.set({
          command_id: `cmd_err_${Date.now()}`,
          device_id: prod.device_id,
          status: 'error',
          error: err.message || 'Execution error or backend unreachable',
        });
      }
    });
  }

  public getBatteryColor(pct?: number): string {
    if (pct == null) return '#94a3b8';
    if (pct > 50) return '#10b981';
    if (pct > 20) return '#f59e0b';
    return '#ef4444';
  }

  public formatFreshness(ts?: number): string {
    if (!ts) return 'Active';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 5) return 'Live (<5s)';
    return `${diff}s ago`;
  }

  public formatResultPayload(res: DeviceCommandResult): string {
    return JSON.stringify(res.result || res.error || res, null, 2);
  }
}
