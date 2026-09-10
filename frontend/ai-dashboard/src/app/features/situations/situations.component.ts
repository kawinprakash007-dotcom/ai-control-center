import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AppStateService } from '../../core/state/app-state.service';
import { Situation, SituationSeverity, SituationStatus } from '../../core/models';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-situations',
  standalone: true,
  imports: [
    CommonModule,
    StatusBadgeComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="situations-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Situation Intelligence Feed</h1>
          <p class="page-subtitle">Multi-product spatial, visual, and telemetry perception correlation engine</p>
        </div>
        <div class="stats-pills">
          <span class="stat-chip critical" [class.active]="appState.criticalSituationsCount() > 0">
            {{ appState.criticalSituationsCount() }} Critical Alerts
          </span>
          <span class="stat-chip">
            {{ situations().length }} Total Situations
          </span>
        </div>
      </div>

      <!-- Filters & Toolbar -->
      <div class="filter-bar">
        <div class="filter-group">
          <span class="filter-label">Severity:</span>
          <button class="filter-btn" [class.active]="selectedSeverity() === 'ALL'" (click)="selectedSeverity.set('ALL')">All</button>
          <button class="filter-btn" [class.active]="selectedSeverity() === 'CRITICAL'" (click)="selectedSeverity.set('CRITICAL')">Critical</button>
          <button class="filter-btn" [class.active]="selectedSeverity() === 'HIGH'" (click)="selectedSeverity.set('HIGH')">High</button>
          <button class="filter-btn" [class.active]="selectedSeverity() === 'MEDIUM'" (click)="selectedSeverity.set('MEDIUM')">Medium</button>
          <button class="filter-btn" [class.active]="selectedSeverity() === 'LOW'" (click)="selectedSeverity.set('LOW')">Low</button>
        </div>

        <button class="btn btn-secondary btn-sm" (click)="appState.refreshWorldState()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          Sync World State
        </button>
      </div>

      <!-- Situations Grid / List -->
      @if (filteredSituations().length === 0) {
        <app-empty-state
          icon="check"
          title="Nominal Operational Envelope"
          message="No active multi-product anomalies, hazards, or perimeter violations detected."
          actionText="Sync Feeds"
          (action)="appState.refreshWorldState()">
        </app-empty-state>
      } @else {
        <div class="situations-grid">
          @for (sit of filteredSituations(); track sit.situation_id) {
            <div class="situation-card card" [class.critical]="sit.severity === 'CRITICAL'">
              <!-- Top Row -->
              <div class="card-header">
                <div class="title-col">
                  <div class="meta-row">
                    <span class="sit-id font-mono">{{ sit.situation_id }}</span>
                    <span class="conf-pill font-mono">{{ (sit.confidence * 100).toFixed(0) }}% CONF</span>
                    <span class="time-meta font-mono">{{ formatTimestamp(sit.detected_at) }}</span>
                  </div>
                  <h3 class="sit-title">{{ sit.title }}</h3>
                </div>

                <div class="status-col">
                  <app-status-badge [status]="sit.severity" size="sm"></app-status-badge>
                  <app-status-badge [status]="sit.status" size="sm"></app-status-badge>
                </div>
              </div>

              <!-- Description -->
              <p class="sit-desc">{{ sit.description }}</p>

              <!-- Participating Products -->
              <div class="section-block">
                <span class="block-label">PARTICIPATING EDGE NODES</span>
                <div class="nodes-row">
                  @for (prodId of sit.participating_products; track prodId) {
                    <span class="node-chip font-mono">
                      <span class="node-dot"></span>
                      {{ prodId }}
                    </span>
                  }
                </div>
              </div>

              <!-- Evidence List -->
              @if (sit.evidence && sit.evidence.length > 0) {
                <div class="section-block">
                  <span class="block-label">CORRELATED PERCEPTION EVIDENCE ({{ sit.evidence.length }})</span>
                  <div class="evidence-list">
                    @for (ev of sit.evidence; track ev.evidence_id) {
                      <div class="evidence-item">
                        <span class="ev-modality font-mono" [ngClass]="ev.modality.toLowerCase()">{{ ev.modality }}</span>
                        <span class="ev-src font-mono">{{ ev.source_device_id }}</span>
                        <span class="ev-desc">{{ ev.description }}</span>
                        <span class="ev-conf font-mono">{{ ((ev.confidence ?? 0.9) * 100).toFixed(0) }}%</span>
                      </div>
                    }
                  </div>
                </div>
              }

              <!-- Recommended Mitigation Action -->
              @if (sit.recommended_action) {
                <div class="action-block">
                  <div class="rec-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00f0ff" stroke-width="2">
                      <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                    </svg>
                    <span class="rec-title">RECOMMENDED COGNITIVE ACTION:</span>
                  </div>
                  <p class="rec-text">{{ sit.recommended_action }}</p>
                </div>
              }
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .situations-container {
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
      gap: 0.75rem;
    }

    .stat-chip {
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
    }

    .stat-chip.critical.active {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.4);
      color: #ef4444;
    }

    .filter-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0f172a;
      padding: 0.6rem 1rem;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .filter-group {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .filter-label {
      font-size: 0.75rem;
      color: #64748b;
      font-weight: 600;
      text-transform: uppercase;
      margin-right: 0.25rem;
    }

    .filter-btn {
      background: transparent;
      border: 1px solid transparent;
      color: #94a3b8;
      font-size: 0.8rem;
      padding: 0.25rem 0.65rem;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .filter-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }

    .filter-btn.active {
      background: rgba(0, 240, 255, 0.12);
      border-color: rgba(0, 240, 255, 0.35);
      color: #00f0ff;
      font-weight: 600;
    }

    .situations-grid {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .situation-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      transition: border-color 0.2s;
    }

    .situation-card.critical {
      border-left: 4px solid #ef4444;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
    }

    .meta-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    }

    .sit-id {
      font-size: 0.75rem;
      color: #64748b;
    }

    .conf-pill {
      font-size: 0.65rem;
      background: rgba(0, 240, 255, 0.1);
      color: #00f0ff;
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      border: 1px solid rgba(0, 240, 255, 0.25);
    }

    .time-meta {
      font-size: 0.7rem;
      color: #475569;
    }

    .sit-title {
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
    }

    .status-col {
      display: flex;
      gap: 0.4rem;
    }

    .sit-desc {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.4;
    }

    .section-block {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .block-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
    }

    .nodes-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .node-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.75rem;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #e2e8f0;
    }

    .node-dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: #00f0ff;
    }

    .evidence-list {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      background: rgba(0, 0, 0, 0.2);
      padding: 0.5rem;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.03);
    }

    .evidence-item {
      display: grid;
      grid-template-columns: 80px 140px 1fr 60px;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.75rem;
    }

    .ev-modality {
      font-size: 0.65rem;
      font-weight: 700;
      padding: 0.1rem 0.3rem;
      border-radius: 3px;
      text-align: center;
    }

    .ev-modality.visual { background: rgba(0, 240, 255, 0.15); color: #00f0ff; }
    .ev-modality.spatial { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .ev-modality.telemetry { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .ev-modality.temporal { background: rgba(16, 185, 129, 0.15); color: #10b981; }

    .ev-src { color: #94a3b8; font-size: 0.7rem; }
    .ev-desc { color: #e2e8f0; }
    .ev-conf { color: #64748b; text-align: right; }

    .action-block {
      background: rgba(0, 240, 255, 0.05);
      border: 1px solid rgba(0, 240, 255, 0.2);
      border-radius: 6px;
      padding: 0.75rem;
    }

    .rec-header {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 0.3rem;
    }

    .rec-title {
      font-size: 0.7rem;
      font-weight: 700;
      color: #00f0ff;
      letter-spacing: 0.05em;
    }

    .rec-text {
      font-size: 0.8rem;
      color: #e2e8f0;
      line-height: 1.4;
    }
  `]
})
export class SituationsComponent {
  public appState = inject(AppStateService);
  public selectedSeverity = signal<'ALL' | SituationSeverity>('ALL');

  // Situations derived from worldState or fallback scenario triggers
  public situations = computed<Situation[]>(() => {
    const ws = this.appState.worldState();
    const list: Situation[] = [];

    // Parse active conditions from WorldState
    if (ws && ws.conditions) {
      if (Array.isArray(ws.conditions)) {
        ws.conditions.forEach((c: any, idx: number) => {
          const valStr = String(c.value ?? '');
          list.push({
            situation_id: `SIT-ENV-${idx + 1}`,
            title: `Environmental Condition: ${(c.property_name || c.condition_id || 'Alert').toUpperCase()}`,
            severity: (valStr === 'alert' || valStr === 'critical') ? 'CRITICAL' : 'MEDIUM',
            status: 'ACTIVE',
            confidence: c.confidence ?? 0.92,
            detected_at: c.timestamp || (Date.now() / 1000 - 120),
            description: `Authoritative WorldState condition report: [${c.property_name} = ${valStr}]`,
            participating_products: ['ATLAS_VISION_01', 'ATLAS_DRONE_01'],
            evidence: [
              {
                evidence_id: `ev_c_${idx}`,
                modality: 'TELEMETRY',
                source_device_id: 'ATLAS_VISION_01',
                timestamp: Date.now() / 1000 - 120,
                confidence: 0.95,
                description: `Sensor threshold condition '${c.property_name}'`,
              }
            ],
            recommended_action: 'Dispatch ATLAS_ROVER_01 to verify perimeter coordinates.',
          });
        });
      }
    }

    // Default canonical tactical situations for full presentation coverage
    if (list.length === 0) {
      list.push(
        {
          situation_id: 'SIT-001-ALPHA',
          title: 'Perimeter Breach & Visual Target Correlation',
          severity: 'CRITICAL',
          status: 'ACTIVE',
          confidence: 0.96,
          detected_at: Date.now() / 1000 - 45,
          description: 'Multi-node visual detection confirmed by Vision Alpha and Drone Sentinel at Sector 4 Gate with high-velocity motion.',
          participating_products: ['ATLAS_VISION_01', 'ATLAS_DRONE_01'],
          evidence: [
            {
              evidence_id: 'ev-v-101',
              modality: 'VISUAL',
              source_device_id: 'ATLAS_VISION_01',
              timestamp: Date.now() / 1000 - 45,
              confidence: 0.98,
              description: 'BoundingBox [0.24, 0.42, 0.38, 0.65] classified as intruder',
            },
            {
              evidence_id: 'ev-s-102',
              modality: 'SPATIAL',
              source_device_id: 'ATLAS_DRONE_01',
              timestamp: Date.now() / 1000 - 40,
              confidence: 0.94,
              description: 'Radar echo at (x=14.2m, y=8.6m, z=0.0m) confirming proximity',
            },
          ],
          recommended_action: 'Direct ATLAS_ROVER_01 to intercept and display warning HUD overlay via ATLAS_GLASS_01.',
        },
        {
          situation_id: 'SIT-002-BRAVO',
          title: 'Thermal Anomaly & Low-Bandwidth Occlusion',
          severity: 'HIGH',
          status: 'INVESTIGATING',
          confidence: 0.89,
          detected_at: Date.now() / 1000 - 180,
          description: 'Thermal hotspot detected in Zone B during aerial sweep. Vision camera obstructed by smoke.',
          participating_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
          evidence: [
            {
              evidence_id: 'ev-t-201',
              modality: 'TELEMETRY',
              source_device_id: 'ATLAS_DRONE_01',
              timestamp: Date.now() / 1000 - 180,
              confidence: 0.91,
              description: 'Infrared sensor readout exceeded 65.0°C at grid (22.5, 41.2)',
            }
          ],
          recommended_action: 'Initiate thermal sweep protocol and deploy ground rover with environmental sensors.',
        }
      );
    }

    return list;
  });

  public filteredSituations = computed(() => {
    const list = this.situations();
    const sev = this.selectedSeverity();
    if (sev === 'ALL') return list;
    return list.filter(s => s.severity === sev);
  });

  public formatTimestamp(ts?: number): string {
    if (!ts) return 'Just now';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }
}
