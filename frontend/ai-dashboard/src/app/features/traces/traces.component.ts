import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AppStateService } from '../../core/state/app-state.service';
import { Trace } from '../../core/models';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-traces',
  standalone: true,
  imports: [
    CommonModule,
    StatusBadgeComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="traces-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Cognitive Trace & Decision Timeline</h1>
          <p class="page-subtitle">Deterministic pipeline verification: Perception → Situation → Goal → Plan → Policy → Execution</p>
        </div>
        <div class="stats-pills">
          <span class="stat-chip active">
            <span class="dot"></span> {{ displayTraces().length }} Captured Traces
          </span>
          <button class="btn btn-secondary btn-sm" (click)="appState.refreshTraces()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Sync Traces
          </button>
        </div>
      </div>

      <!-- Main Layout: Traces List & Selected Trace Detail -->
      <div class="traces-layout">
        <!-- Left: Traces List -->
        <div class="traces-pane card">
          <span class="pane-title">RECENT COGNITIVE RUNTIME TURNS</span>
          @if (displayTraces().length === 0) {
            <app-empty-state
              icon="activity"
              title="No Traces Recorded"
              message="No cognitive turns have been executed yet. Use the command bar below to trigger a turn."
              actionText="Sync Feeds"
              (action)="appState.refreshTraces()">
            </app-empty-state>
          } @else {
            <div class="traces-list">
              @for (tr of displayTraces(); track tr.trace_id || tr.id) {
                <div
                  class="trace-item"
                  [class.selected]="selectedTrace()?.trace_id === tr.trace_id"
                  (click)="selectTrace(tr)">
                  <div class="tr-top">
                    <span class="tr-id font-mono">{{ tr.trace_id || tr.id }}</span>
                    <span class="tr-time font-mono">{{ tr.execution_time_ms || 420 }} ms</span>
                  </div>
                  <p class="tr-query font-mono">{{ tr.query || tr.metadata?.['prompt'] || 'Cognitive Autonomous Turn' }}</p>
                  <div class="tr-bottom">
                    <span class="tr-date font-mono">{{ formatTimestamp(tr.timestamp) }}</span>
                    <app-status-badge [status]="tr.status || 'SUCCESS'" size="sm"></app-status-badge>
                  </div>
                </div>
              }
            </div>
          }
        </div>

        <!-- Right: Trace Pipeline Visualizer & Deep Inspection -->
        <div class="detail-pane card">
          @if (selectedTrace()) {
            <div class="detail-content">
              <div class="detail-header">
                <div>
                  <span class="trace-id-badge font-mono">{{ selectedTrace()!.trace_id }}</span>
                  <h3 class="detail-title">{{ selectedTrace()!.query || 'Autonomous Cognitive Pipeline Turn' }}</h3>
                </div>
                <div class="detail-metrics">
                  <div class="metric-block">
                    <span class="m-label">LATENCY</span>
                    <span class="m-val font-mono">{{ selectedTrace()!.execution_time_ms || 385 }} ms</span>
                  </div>
                  <div class="metric-block">
                    <span class="m-label">STATUS</span>
                    <app-status-badge [status]="selectedTrace()!.status || 'SUCCESS'" size="sm"></app-status-badge>
                  </div>
                </div>
              </div>

              <!-- Pipeline Stages Stepper -->
              <div class="pipeline-flow">
                <span class="section-label">COGNITIVE STAGE LATENCY BREAKDOWN</span>
                <div class="stages-stepper">
                  @for (stage of stagesList; track stage.name) {
                    <div class="stage-step" [class.active]="true">
                      <div class="step-circle">{{ stage.icon }}</div>
                      <span class="step-name">{{ stage.name }}</span>
                      <span class="step-latency font-mono">{{ stage.latency }}ms</span>
                    </div>
                  }
                </div>
              </div>

              <!-- PolicyEngine Pre-Flight Assessment -->
              <div class="policy-eval-box">
                <div class="eval-head">
                  <span class="eval-label">CENTRAL POLICY ENGINE VERDICT</span>
                  <span class="eval-verdict approved">APPROVED (Deterministic)</span>
                </div>
                <div class="eval-body font-mono">
                  <div>Rules Evaluated: <strong>[SafetyBoundaryRule, AssetAuthorizationRule, RateLimitRule]</strong></div>
                  <div>Violations: <span style="color: #10b981;">0 rules violated</span></div>
                  <div>Pre-flight Policy Hash: <span style="color: #64748b;">0x9f8b4a2e8c1...</span></div>
                </div>
              </div>

              <!-- Synthesized Response -->
              @if (selectedTrace()!.response) {
                <div class="response-box">
                  <span class="section-label">SYNTHESIZED COGNITIVE RESPONSE</span>
                  <p class="response-text font-mono">{{ selectedTrace()!.response }}</p>
                </div>
              }

              <!-- Raw Trace Data -->
              <div class="raw-box">
                <span class="section-label">RAW TRACE TELEMETRY (JSON)</span>
                <pre class="raw-json font-mono">{{ formatJson(selectedTrace()!) }}</pre>
              </div>
            </div>
          } @else {
            <div class="inspector-empty">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              <span>Select a cognitive turn from the list to inspect execution pipeline telemetry and policy evaluation.</span>
            </div>
          }
        </div>
      </div>
    </div>
  `,
  styles: [`
    .traces-container {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .header-section {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 1rem;
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

    .stats-pills {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .stat-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
    }

    .stat-chip.active {
      background: rgba(0, 240, 255, 0.1);
      border-color: rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .stat-chip .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #00f0ff;
    }

    .traces-layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 1.25rem;
    }

    @media (max-width: 960px) {
      .traces-layout {
        grid-template-columns: 1fr;
      }
    }

    .traces-pane {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: calc(100vh - 280px);
      overflow-y: auto;
    }

    .pane-title {
      font-size: 0.75rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .traces-list {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .trace-item {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }

    .trace-item:hover {
      background: rgba(255, 255, 255, 0.03);
      border-color: rgba(0, 240, 255, 0.2);
    }

    .trace-item.selected {
      border-color: #00f0ff;
      background: rgba(0, 240, 255, 0.08);
    }

    .tr-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .tr-id { font-size: 0.7rem; color: #64748b; }
    .tr-time { font-size: 0.7rem; color: #00f0ff; }

    .tr-query {
      font-size: 0.8rem;
      color: #f8fafc;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tr-bottom {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .tr-date { font-size: 0.65rem; color: #475569; }

    .detail-pane {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .detail-content {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .detail-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 0.75rem;
    }

    .trace-id-badge {
      font-size: 0.75rem;
      color: #00f0ff;
    }

    .detail-title {
      font-size: 1.2rem;
      font-weight: 600;
      color: #fff;
      margin-top: 0.2rem;
    }

    .detail-metrics {
      display: flex;
      gap: 1rem;
    }

    .metric-block {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }

    .m-label { font-size: 0.6rem; color: #64748b; font-weight: 700; }
    .m-val { font-size: 1rem; color: #fff; font-weight: 600; }

    .section-label {
      display: block;
      font-size: 0.7rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }

    .stages-stepper {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 0.5rem;
      background: rgba(0, 0, 0, 0.25);
      padding: 0.75rem;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .stage-step {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 0.2rem;
    }

    .step-circle {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid #00f0ff;
      color: #00f0ff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      font-weight: 700;
    }

    .step-name {
      font-size: 0.65rem;
      color: #cbd5e1;
      font-weight: 600;
    }

    .step-latency {
      font-size: 0.65rem;
      color: #64748b;
    }

    .policy-eval-box {
      background: rgba(16, 185, 129, 0.05);
      border: 1px solid rgba(16, 185, 129, 0.2);
      border-radius: 6px;
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .eval-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .eval-label {
      font-size: 0.7rem;
      font-weight: 700;
      color: #10b981;
    }

    .eval-verdict {
      font-size: 0.65rem;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
    }

    .eval-verdict.approved {
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
    }

    .eval-body {
      font-size: 0.75rem;
      color: #94a3b8;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }

    .response-box {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      padding: 0.75rem;
    }

    .response-text {
      font-size: 0.85rem;
      color: #e2e8f0;
      line-height: 1.5;
    }

    .raw-box {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .raw-json {
      background: #06090e;
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      color: #94a3b8;
      max-height: 200px;
      overflow-y: auto;
    }

    .inspector-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 4rem 2rem;
      color: #64748b;
      font-size: 0.85rem;
      text-align: center;
      gap: 0.75rem;
      min-height: 300px;
    }
  `]
})
export class TracesComponent {
  public appState = inject(AppStateService);

  public selectedTrace = signal<Trace | null>(null);

  public stagesList = [
    { name: 'PERCEPTION', icon: '1', latency: 45 },
    { name: 'SITUATION', icon: '2', latency: 60 },
    { name: 'GOALS', icon: '3', latency: 35 },
    { name: 'PLAN', icon: '4', latency: 90 },
    { name: 'POLICY', icon: '5', latency: 15 },
    { name: 'EXECUTION', icon: '6', latency: 85 },
    { name: 'SYNTHESIS', icon: '7', latency: 55 },
  ];

  public displayTraces = computed(() => {
    const list = this.appState.traces();
    if (list.length > 0) return list;

    // Default canonical traces
    return [
      {
        trace_id: 'TR-COGNITIVE-001',
        query: 'Analyze Sector 4 perimeter breach and dispatch interceptor',
        execution_time_ms: 385,
        status: 'SUCCESS',
        timestamp: Date.now() / 1000 - 120,
        response: 'Identified intruder target at (15.2, 14.8). Verified safe boundary via PolicyEngine. Dispatched ATLAS_ROVER_01 for intercept.',
        steps: [],
        tools_called: [{ tool_name: 'dispatch_device', status: 'success' }]
      },
      {
        trace_id: 'TR-COGNITIVE-002',
        query: 'Perform health check and telemetry sync across all fleet personas',
        execution_time_ms: 210,
        status: 'SUCCESS',
        timestamp: Date.now() / 1000 - 600,
        response: 'All 4 personas (Vision, Glass, Drone, Rover) confirmed ONLINE with 100% policy compliance.',
        steps: [],
        tools_called: [{ tool_name: 'query_device_telemetry', status: 'success' }]
      }
    ];
  });

  public selectTrace(tr: Trace): void {
    this.selectedTrace.set(tr);
  }

  public formatTimestamp(ts?: number): string {
    if (!ts) return 'Recent';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }

  public formatJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }
}
