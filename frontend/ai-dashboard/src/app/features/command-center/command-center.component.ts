import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AppStateService } from '../../core/state/app-state.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { SpatialCanvasComponent } from '../../shared/components/spatial-canvas/spatial-canvas.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';
import { Product } from '../../core/models';

@Component({
  selector: 'app-command-center',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    StatusBadgeComponent,
    SpatialCanvasComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="dashboard-page">
      <!-- Section A: Top System KPI Row -->
      <div class="kpi-grid">
        <div class="kpi-card card">
          <div class="kpi-label mono text-xs">CENTRAL RUNTIME</div>
          <div class="kpi-val-row">
            <span class="kpi-val text-xl font-bold mono" [class.text-healthy]="state.systemHealth()?.status === 'healthy'">
              {{ (state.systemHealth()?.status || 'CONNECTING') | uppercase }}
            </span>
            <app-status-badge [status]="state.systemHealth()?.status || 'UNKNOWN'" [pulse]="true"></app-status-badge>
          </div>
          <div class="kpi-sub mono text-xs text-dim">
            ENV: {{ state.systemHealth()?.app_env || 'production' }} | SIM: {{ state.systemReady()?.simulation_mode ? 'ENABLED' : 'DISABLED' }}
          </div>
        </div>

        <div class="kpi-card card">
          <div class="kpi-label mono text-xs">ACTIVE PRODUCTS</div>
          <div class="kpi-val-row">
            <span class="kpi-val text-xl font-bold mono text-accent">
              {{ state.onlineProductsCount() }} / {{ state.products().length }}
            </span>
            <span class="mono text-xs text-secondary">ONLINE</span>
          </div>
          <div class="kpi-sub mono text-xs text-dim">
            VISION • GLASS • DRONE • ROVER
          </div>
        </div>

        <div class="kpi-card card">
          <div class="kpi-label mono text-xs">WORLD ENTITIES</div>
          <div class="kpi-val-row">
            <span class="kpi-val text-xl font-bold mono text-healthy">
              {{ state.worldState()?.entities?.length || 0 }}
            </span>
            <span class="mono text-xs text-dim">V{{ state.worldState()?.version || 1 }}</span>
          </div>
          <div class="kpi-sub mono text-xs text-dim">
            CONDITIONS: {{ state.worldState()?.conditions?.length || 0 }} | RELS: {{ state.worldState()?.relationships?.length || 0 }}
          </div>
        </div>

        <div class="kpi-card card">
          <div class="kpi-label mono text-xs">ACTIVE GOALS</div>
          <div class="kpi-val-row">
            <span class="kpi-val text-xl font-bold mono text-warning">
              {{ state.activeGoalsCount() }}
            </span>
            <span class="mono text-xs text-dim">IN PROGRESS</span>
          </div>
          <div class="kpi-sub mono text-xs text-dim">
            TOTAL TRACKED: {{ state.goals().length }}
          </div>
        </div>
      </div>

      <!-- Section B: Central Spatial Radar & 4 Products Status -->
      <div class="main-split-grid">
        <div class="spatial-column">
          <app-spatial-canvas
            [products]="state.products()"
            [situations]="state.activeSituations()"
            (productSelected)="onSelectProduct($event)"
          ></app-spatial-canvas>
        </div>

        <!-- Four Product Status Cards -->
        <div class="products-column">
          <div class="column-header">
            <span class="mono text-xs text-accent">ATLAS PRODUCT PERSONAS</span>
            <a routerLink="/products" class="mono text-xs text-dim hover-link">VIEW ALL →</a>
          </div>

          <div class="product-cards-stack">
            <div *ngFor="let p of state.products()" class="product-mini-card card" [routerLink]="['/products', p.device_id]">
              <div class="p-header">
                <div class="p-title-wrap">
                  <span class="p-icon">{{ getProductIcon(p.product_type) }}</span>
                  <div>
                    <div class="p-name font-semibold text-sm">{{ p.display_name }}</div>
                    <div class="p-role mono text-xs text-dim">{{ p.product_role }}</div>
                  </div>
                </div>
                <app-status-badge [status]="p.connectivity_status"></app-status-badge>
              </div>

              <div class="p-metrics mono text-xs">
                <div class="metric-item">
                  <span class="m-lbl text-dim">BATTERY:</span>
                  <span class="m-val" [class.text-warning]="(p.telemetry?.battery_level || 100) < 20">
                    {{ p.telemetry?.battery_level || 100 | number:'1.0-0' }}%
                  </span>
                </div>
                <div class="metric-item">
                  <span class="m-lbl text-dim">HEALTH:</span>
                  <span class="m-val" [class.text-healthy]="p.health_status === 'HEALTHY'">
                    {{ p.health_status }}
                  </span>
                </div>
                <div *ngIf="p.telemetry?.speed_mps" class="metric-item">
                  <span class="m-lbl text-dim">SPEED:</span>
                  <span class="m-val">{{ p.telemetry?.speed_mps }} m/s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Section C: Active Goals & Real-Time Event Feed -->
      <div class="bottom-split-grid">
        <!-- Active Goals -->
        <div class="panel card">
          <div class="panel-header">
            <span class="mono text-xs text-accent">ACTIVE TACTICAL GOALS</span>
            <span class="mono text-xs text-dim">{{ state.goals().length }} TOTAL</span>
          </div>

          <div *ngIf="state.goals().length === 0" class="panel-body">
            <app-empty-state
              icon="🎯"
              title="No Active Goals"
              message="No goals currently in progress. Commands submitted via 'ASK ATLAS' will generate tactical goals."
            ></app-empty-state>
          </div>

          <div *ngIf="state.goals().length > 0" class="panel-body list-scroll">
            <div *ngFor="let g of state.goals()" class="goal-item card">
              <div class="goal-top">
                <span class="goal-title font-semibold text-sm">{{ g.original_goal || g.goal }}</span>
                <app-status-badge [status]="g.status"></app-status-badge>
              </div>
              <div class="goal-meta mono text-xs text-dim">
                <span>PRIORITY: {{ g.priority }}</span>
                <span *ngIf="g.progress?.percentage !== undefined">PROGRESS: {{ g.progress?.percentage | number:'1.0-0' }}%</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Real-Time Cognitive Events Feed -->
        <div class="panel card">
          <div class="panel-header">
            <span class="mono text-xs text-accent">REAL-TIME COGNITIVE & TELEMETRY STREAM</span>
            <span class="mono text-xs text-dim">{{ state.events().length }} EVENTS</span>
          </div>

          <div *ngIf="state.events().length === 0" class="panel-body">
            <app-empty-state
              icon="⚡"
              title="Listening for Events"
              message="WebSocket stream connected to ATLAS Central. Cognitive turns and telemetry updates will stream here live."
            ></app-empty-state>
          </div>

          <div *ngIf="state.events().length > 0" class="panel-body list-scroll">
            <div *ngFor="let ev of state.events()" class="event-item">
              <span class="ev-dot"></span>
              <div class="ev-content">
                <div class="ev-row">
                  <span class="ev-type mono text-xs font-semibold text-accent">{{ ev.event_type }}</span>
                  <span class="ev-time mono text-xs text-dim">{{ ev.timestamp * 1000 | date:'HH:mm:ss' }}</span>
                </div>
                <div class="ev-msg text-xs text-secondary">{{ ev.message }}</div>
                <div *ngIf="ev.correlation_id" class="ev-corr mono text-xs text-dim">TURN: {{ ev.correlation_id }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .dashboard-page {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    @media (max-width: 1280px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .kpi-label { color: var(--text-dim); }
    .kpi-val-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .kpi-sub { opacity: 0.7; }

    .main-split-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
    }
    @media (max-width: 1200px) {
      .main-split-grid { grid-template-columns: 1fr; }
    }
    .products-column {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .column-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 4px 0;
    }
    .hover-link:hover { color: var(--accent-cyan); }
    .product-cards-stack {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .product-mini-card {
      padding: 12px 14px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .product-mini-card:hover {
      border-color: var(--border-accent);
      transform: translateY(-1px);
    }
    .p-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .p-title-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .p-icon { font-size: 18px; }
    .p-metrics {
      display: flex;
      align-items: center;
      gap: 14px;
      border-top: 1px solid rgba(255, 255, 255, 0.04);
      padding-top: 6px;
    }
    .metric-item {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .bottom-split-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 1024px) {
      .bottom-split-grid { grid-template-columns: 1fr; }
    }
    .panel {
      display: flex;
      flex-direction: column;
      max-height: 380px;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(0, 0, 0, 0.2);
    }
    .panel-body {
      padding: 14px 16px;
      overflow-y: auto;
      flex: 1;
    }
    .list-scroll {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .goal-item {
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .goal-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .goal-meta {
      display: flex;
      gap: 14px;
    }

    .event-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .event-item:last-child { border-bottom: none; }
    .ev-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent-cyan);
      margin-top: 6px;
    }
    .ev-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .ev-row {
      display: flex;
      justify-content: space-between;
    }
  `]
})
export class CommandCenterComponent {
  constructor(public state: AppStateService) {}

  onSelectProduct(p: Product): void {
    // Spatial selection handler
  }

  getProductIcon(type: string): string {
    switch (type) {
      case 'VISION': return '📷';
      case 'GLASS': return '👓';
      case 'DRONE': return '🛸';
      case 'ROVER': return '🚙';
      default: return '🤖';
    }
  }
}
