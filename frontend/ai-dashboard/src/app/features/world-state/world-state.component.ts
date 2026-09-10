import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../core/state/app-state.service';
import { WorldEntity, WorldCondition } from '../../core/models';
import { SpatialCanvasComponent } from '../../shared/components/spatial-canvas/spatial-canvas.component';

@Component({
  selector: 'app-world-state',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    SpatialCanvasComponent,
  ],
  template: `
    <div class="world-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">World State & Digital Twin</h1>
          <p class="page-subtitle">Authoritative ground-truth spatial representation, entity tracking, and environmental models</p>
        </div>
        <div class="stats-pills">
          <span class="stat-chip active">
            <span class="dot"></span> {{ entities().length }} Spatial Entities
          </span>
          <span class="stat-chip">
            {{ conditionsList().length }} Active Conditions
          </span>
        </div>
      </div>

      <!-- Main Layout: 2D Radar Canvas & Entity Table -->
      <div class="world-layout">
        <!-- Left: Spatial Canvas -->
        <div class="spatial-pane card">
          <div class="pane-header">
            <span class="pane-title">DIGITAL TWIN RADAR & SPATIAL CANVAS</span>
            <span class="range-tag">Range: ±50m Local UTM</span>
          </div>
          <div class="canvas-wrapper">
            <app-spatial-canvas
              [products]="appState.products()"
              [entities]="entities()"
              [selectedEntity]="selectedEntity()"
              (entitySelected)="selectEntity($event)">
            </app-spatial-canvas>
          </div>
        </div>

        <!-- Right: Environmental Conditions & Selected Entity Inspection -->
        <div class="side-pane">
          <!-- Environmental Conditions Card -->
          <div class="card conditions-card">
            <h3 class="pane-title">ENVIRONMENTAL ATTRIBUTES</h3>
            <div class="conditions-grid">
              @for (cond of conditionsList(); track cond.key) {
                <div class="cond-cell">
                  <span class="c-key">{{ cond.key | uppercase }}</span>
                  <span class="c-val font-mono">{{ cond.val }}</span>
                </div>
              }
              @if (conditionsList().length === 0) {
                <div class="empty-conditions">No special environmental conditions flagged</div>
              }
            </div>
          </div>

          <!-- Entity Detail Inspector -->
          <div class="card inspector-card">
            <h3 class="pane-title">ENTITY INSPECTION</h3>
            @if (selectedEntity()) {
              <div class="inspector-body">
                <div class="ent-header">
                  <div>
                    <span class="ent-id font-mono">{{ selectedEntity()!.entity_id }}</span>
                    <h4 class="ent-type">{{ selectedEntity()!.name }} ({{ selectedEntity()!.entity_type }})</h4>
                  </div>
                  <span class="freshness-badge font-mono" [ngClass]="getFreshnessClass(selectedEntity()!.last_updated)">
                    {{ getFreshnessLabel(selectedEntity()!.last_updated) }}
                  </span>
                </div>

                <div class="detail-row">
                  <span class="d-label">COORDINATES:</span>
                  <span class="d-val font-mono">
                    Lat: {{ selectedEntity()!.location?.latitude?.toFixed(4) ?? '0.0000' }},
                    Lon: {{ selectedEntity()!.location?.longitude?.toFixed(4) ?? '0.0000' }},
                    Alt: {{ selectedEntity()!.location?.altitude?.toFixed(1) ?? '0.0' }}m
                  </span>
                </div>

                <div class="detail-row">
                  <span class="d-label">CONFIDENCE:</span>
                  <span class="d-val font-mono">{{ ((selectedEntity()!.confidence ?? 0.9) * 100).toFixed(0) }}%</span>
                </div>

                <div class="detail-row">
                  <span class="d-label">LAST OBSERVED:</span>
                  <span class="d-val font-mono">{{ formatTimestamp(selectedEntity()!.last_updated) }}</span>
                </div>

                <div class="attrs-box">
                  <span class="attrs-label">PROPERTIES & ATTRIBUTES:</span>
                  <pre class="attrs-json font-mono">{{ formatJson(selectedEntity()!.properties) }}</pre>
                </div>
              </div>
            } @else {
              <div class="inspector-empty">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="16" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span>Click an entity in the radar or table below to inspect attributes.</span>
              </div>
            }
          </div>
        </div>
      </div>

      <!-- Entities Table Card -->
      <div class="card table-card">
        <div class="table-header">
          <div class="filter-group">
            <span class="f-label">Filter Entities:</span>
            <button class="filter-btn" [class.active]="entityTypeFilter() === 'ALL'" (click)="entityTypeFilter.set('ALL')">All</button>
            <button class="filter-btn" [class.active]="entityTypeFilter() === 'OBSTACLE'" (click)="entityTypeFilter.set('OBSTACLE')">Obstacles</button>
            <button class="filter-btn" [class.active]="entityTypeFilter() === 'PERSON'" (click)="entityTypeFilter.set('PERSON')">Persons</button>
            <button class="filter-btn" [class.active]="entityTypeFilter() === 'VEHICLE'" (click)="entityTypeFilter.set('VEHICLE')">Vehicles</button>
            <button class="filter-btn" [class.active]="entityTypeFilter() === 'HAZARD'" (click)="entityTypeFilter.set('HAZARD')">Hazards</button>
          </div>
          <button class="btn btn-secondary btn-sm" (click)="appState.refreshWorldState()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Refresh State
          </button>
        </div>

        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>ENTITY ID</th>
                <th>NAME</th>
                <th>TYPE</th>
                <th>LOCATION</th>
                <th>CONFIDENCE</th>
                <th>FRESHNESS</th>
                <th>LAST OBSERVED</th>
                <th>ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              @for (ent of filteredEntities(); track ent.entity_id) {
                <tr [class.selected]="selectedEntity()?.entity_id === ent.entity_id" (click)="selectEntity(ent)">
                  <td class="font-mono text-cyan">{{ ent.entity_id }}</td>
                  <td>{{ ent.name }}</td>
                  <td>
                    <span class="type-badge" [ngClass]="ent.entity_type.toLowerCase()">{{ ent.entity_type }}</span>
                  </td>
                  <td class="font-mono">
                    ({{ ent.location?.latitude?.toFixed(3) ?? 0 }}, {{ ent.location?.longitude?.toFixed(3) ?? 0 }})
                  </td>
                  <td class="font-mono">{{ ((ent.confidence ?? 0.9) * 100).toFixed(0) }}%</td>
                  <td>
                    <span class="freshness-chip font-mono" [ngClass]="getFreshnessClass(ent.last_updated)">
                      {{ getFreshnessLabel(ent.last_updated) }}
                    </span>
                  </td>
                  <td class="font-mono text-muted">{{ formatTimestamp(ent.last_updated) }}</td>
                  <td>
                    <button class="btn btn-secondary btn-xs" (click)="selectEntity(ent); $event.stopPropagation()">
                      Inspect
                    </button>
                  </td>
                </tr>
              }
              @if (filteredEntities().length === 0) {
                <tr>
                  <td colspan="8" class="text-center py-4 text-muted">No entities match the filter.</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .world-container {
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

    .world-layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }

    @media (max-width: 900px) {
      .world-layout {
        grid-template-columns: 1fr;
      }
    }

    .spatial-pane {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .pane-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .pane-title {
      font-size: 0.75rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .range-tag {
      font-size: 0.7rem;
      color: #64748b;
      font-family: var(--font-mono);
    }

    .canvas-wrapper {
      display: flex;
      justify-content: center;
      align-items: center;
      background: #06090e;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.5rem;
    }

    .side-pane {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .conditions-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .conditions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.5rem;
    }

    .cond-cell {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }

    .c-key {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
    }

    .c-val {
      font-size: 0.85rem;
      font-weight: 600;
      color: #00f0ff;
    }

    .empty-conditions {
      font-size: 0.8rem;
      color: #475569;
      font-style: italic;
      grid-column: 1 / -1;
    }

    .inspector-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      flex: 1;
    }

    .inspector-body {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .ent-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 0.5rem;
    }

    .ent-id {
      font-size: 0.8rem;
      color: #00f0ff;
    }

    .ent-type {
      font-size: 1.1rem;
      font-weight: 600;
      color: #fff;
    }

    .detail-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    .d-label {
      color: #64748b;
      font-weight: 600;
    }

    .d-val {
      color: #cbd5e1;
    }

    .attrs-box {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      margin-top: 0.25rem;
    }

    .attrs-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
    }

    .attrs-json {
      background: rgba(0, 0, 0, 0.3);
      padding: 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #94a3b8;
      max-height: 120px;
      overflow-y: auto;
    }

    .inspector-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem 1rem;
      color: #64748b;
      font-size: 0.8rem;
      text-align: center;
      gap: 0.5rem;
    }

    .table-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .table-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .filter-group {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .f-label {
      font-size: 0.75rem;
      color: #64748b;
      font-weight: 600;
    }

    .filter-btn {
      background: transparent;
      border: 1px solid transparent;
      color: #94a3b8;
      font-size: 0.75rem;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      cursor: pointer;
    }

    .filter-btn.active {
      background: rgba(0, 240, 255, 0.12);
      border-color: rgba(0, 240, 255, 0.35);
      color: #00f0ff;
    }

    .table-responsive {
      overflow-x: auto;
    }

    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }

    .data-table th {
      text-align: left;
      padding: 0.6rem 0.75rem;
      color: #64748b;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .data-table td {
      padding: 0.6rem 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }

    .data-table tbody tr {
      cursor: pointer;
      transition: background 0.15s;
    }

    .data-table tbody tr:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    .data-table tbody tr.selected {
      background: rgba(0, 240, 255, 0.08);
    }

    .type-badge {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.06);
    }

    .type-badge.obstacle { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .type-badge.person { background: rgba(0, 240, 255, 0.15); color: #00f0ff; }
    .type-badge.vehicle { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .type-badge.hazard { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

    .freshness-chip, .freshness-badge {
      font-size: 0.65rem;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
    }

    .freshness-chip.current, .freshness-badge.current {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
    }

    .freshness-chip.stale, .freshness-badge.stale {
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
    }

    .freshness-chip.expired, .freshness-badge.expired {
      background: rgba(239, 68, 68, 0.15);
      color: #ef4444;
    }

    .text-cyan { color: #00f0ff; }
    .text-muted { color: #64748b; }
    .btn-xs { padding: 0.2rem 0.5rem; font-size: 0.7rem; }
  `]
})
export class WorldStateComponent {
  public appState = inject(AppStateService);

  public selectedEntity = signal<WorldEntity | null>(null);
  public entityTypeFilter = signal<string>('ALL');

  public entities = computed<WorldEntity[]>(() => {
    const ws = this.appState.worldState();
    if (ws && ws.entities && ws.entities.length > 0) {
      return ws.entities;
    }

    // Default canonical digital twin state entities
    return [
      {
        entity_id: 'ENT-GATE-01',
        entity_type: 'OBSTACLE',
        name: 'Sector 4 Perimeter Gate',
        status: 'ACTIVE',
        location: { latitude: 37.7750, longitude: -122.4190, altitude: 0.0 },
        confidence: 0.98,
        last_updated: Date.now() / 1000 - 15,
        properties: { material: 'steel', security_clearance: 'restricted', status: 'closed' }
      },
      {
        entity_id: 'ENT-TARGET-99',
        entity_type: 'PERSON',
        name: 'Unidentified Intruder',
        status: 'ACTIVE',
        location: { latitude: 37.7752, longitude: -122.4188, altitude: 0.0 },
        confidence: 0.94,
        last_updated: Date.now() / 1000 - 8,
        properties: { velocity_mps: 2.1, heading: 45.0, classification: 'unauthorized_visitor' }
      },
      {
        entity_id: 'ENT-CONTAINER-03',
        entity_type: 'HAZARD',
        name: 'Thermal Waste Container 3',
        status: 'ACTIVE',
        location: { latitude: 37.7745, longitude: -122.4200, altitude: 0.0 },
        confidence: 0.91,
        last_updated: Date.now() / 1000 - 120,
        properties: { thermal_signature: 'high', temp_c: 68.4 }
      },
      {
        entity_id: 'ENT-ROVER-ECHO',
        entity_type: 'VEHICLE',
        name: 'ATLAS Rover Vanguard',
        status: 'ACTIVE',
        location: { latitude: 37.7747, longitude: -122.4195, altitude: 0.0 },
        confidence: 0.99,
        last_updated: Date.now() / 1000 - 2,
        properties: { linked_device: 'ATLAS_ROVER_01', mode: 'autonomous_patrol' }
      }
    ];
  });

  public filteredEntities = computed(() => {
    const list = this.entities();
    const filter = this.entityTypeFilter();
    if (filter === 'ALL') return list;
    return list.filter(e => e.entity_type.toUpperCase() === filter);
  });

  public conditionsList = computed(() => {
    const ws = this.appState.worldState();
    const list: { key: string; val: string }[] = [];
    if (ws && ws.conditions && Array.isArray(ws.conditions)) {
      ws.conditions.forEach((c: WorldCondition) => {
        list.push({ key: c.property_name || c.condition_id, val: String(c.value) });
      });
    } else if (ws && ws.conditions && typeof ws.conditions === 'object') {
      Object.entries(ws.conditions).forEach(([k, v]) => {
        list.push({ key: k, val: String(v) });
      });
    }
    if (list.length === 0) {
      return [
        { key: 'Visibility', val: 'Clear (10km)' },
        { key: 'Temperature', val: '22.5°C' },
        { key: 'Wind', val: '3.2 m/s NE' },
        { key: 'GPS Quality', val: 'HDOP 0.8 (Fix 3D)' }
      ];
    }
    return list;
  });

  public selectEntity(ent: WorldEntity): void {
    this.selectedEntity.set(ent);
  }

  public getFreshnessLabel(lastUpdated?: number): string {
    if (!lastUpdated) return 'CURRENT';
    const age = Date.now() / 1000 - lastUpdated;
    if (age < 10) return 'CURRENT';
    if (age < 60) return 'STALE';
    return 'EXPIRED';
  }

  public getFreshnessClass(lastUpdated?: number): string {
    return this.getFreshnessLabel(lastUpdated).toLowerCase();
  }

  public formatTimestamp(ts?: number): string {
    if (!ts) return 'Live';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 5) return 'Just now';
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }

  public formatJson(obj: any): string {
    return JSON.stringify(obj || {}, null, 2);
  }
}
