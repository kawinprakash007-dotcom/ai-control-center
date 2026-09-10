import { Component, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { AppStateService } from '../../core/state/app-state.service';
import { Product, ProductType } from '../../core/models';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { TelemetryChartComponent } from '../../shared/components/telemetry-chart/telemetry-chart.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-products',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    StatusBadgeComponent,
    TelemetryChartComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="products-container fade-in">
      <!-- Header / Toolbar -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Edge & Autonomous Fleet</h1>
          <p class="page-subtitle">Authoritative hardware contracts & live telemetry status across all 4 operational personas</p>
        </div>
        <div class="stats-pills">
          <span class="stat-chip online">
            <span class="dot"></span> {{ appState.onlineProductsCount() }} Active Online
          </span>
          <span class="stat-chip total">
            {{ appState.products().length }} Registered Nodes
          </span>
        </div>
      </div>

      <!-- Filters -->
      <div class="filter-bar">
        <div class="filter-group">
          <span class="filter-label">Filter Type:</span>
          <button
            class="filter-btn"
            [class.active]="selectedType() === 'ALL'"
            (click)="selectedType.set('ALL')">All</button>
          <button
            class="filter-btn"
            [class.active]="selectedType() === 'VISION'"
            (click)="selectedType.set('VISION')">Vision Core</button>
          <button
            class="filter-btn"
            [class.active]="selectedType() === 'GLASS'"
            (click)="selectedType.set('GLASS')">Glass HUD</button>
          <button
            class="filter-btn"
            [class.active]="selectedType() === 'DRONE'"
            (click)="selectedType.set('DRONE')">Drone Sentinel</button>
          <button
            class="filter-btn"
            [class.active]="selectedType() === 'ROVER'"
            (click)="selectedType.set('ROVER')">Rover Vanguard</button>
        </div>

        <button class="btn btn-secondary btn-sm" (click)="appState.refreshDevices()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          Sync Fleet
        </button>
      </div>

      <!-- Product Cards Grid -->
      @if (filteredProducts().length === 0) {
        <app-empty-state
          icon="hardware"
          title="No Products Matched"
          message="No active edge products or simulated vehicles match the selected filter."
          actionText="Reset Filters"
          (action)="selectedType.set('ALL')">
        </app-empty-state>
      } @else {
        <div class="products-grid">
          @for (prod of filteredProducts(); track prod.device_id) {
            <div class="product-card card">
              <!-- Card Header -->
              <div class="card-header">
                <div class="persona-icon-box" [ngClass]="prod.product_type.toLowerCase()">
                  @switch (prod.product_type) {
                    @case ('VISION') {
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                      </svg>
                    }
                    @case ('GLASS') {
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="6" cy="12" r="4"></circle>
                        <circle cx="18" cy="12" r="4"></circle>
                        <line x1="10" y1="12" x2="14" y2="12"></line>
                      </svg>
                    }
                    @case ('DRONE') {
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                      </svg>
                    }
                    @case ('ROVER') {
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect>
                        <line x1="1" y1="10" x2="23" y2="10"></line>
                      </svg>
                    }
                  }
                </div>

                <div class="title-meta">
                  <div class="id-row">
                    <span class="device-id font-mono">{{ prod.device_id }}</span>
                    @if (prod.is_simulation) {
                      <span class="sim-badge">SIMULATED</span>
                    }
                  </div>
                  <h3 class="product-title">{{ prod.display_name }}</h3>
                </div>

                <div class="status-col">
                  <app-status-badge [status]="prod.connectivity_status" size="sm"></app-status-badge>
                </div>
              </div>

              <!-- Metrics Row -->
              <div class="metrics-row">
                <div class="metric-cell">
                  <span class="m-label">ROLE</span>
                  <span class="m-val font-mono">{{ prod.product_role || 'HYBRID' }}</span>
                </div>
                <div class="metric-cell">
                  <span class="m-label">BATTERY</span>
                  <span class="m-val font-mono" [style.color]="getBatteryColor(prod.telemetry?.battery_level)">
                    {{ prod.telemetry?.battery_level != null ? (prod.telemetry?.battery_level + '%') : 'N/A' }}
                  </span>
                </div>
                <div class="metric-cell">
                  <span class="m-label">TEMP</span>
                  <span class="m-val font-mono">{{ prod.telemetry?.temperature_celsius ?? 24 }}°C</span>
                </div>
                <div class="metric-cell">
                  <span class="m-label">SPEED</span>
                  <span class="m-val font-mono">{{ prod.telemetry?.speed_mps ?? 0 }} m/s</span>
                </div>
              </div>

              <!-- Sparkline Trend Chart -->
              <div class="trend-section">
                <span class="section-label">TELEMETRY TREND (BATTERY / LOAD)</span>
                <app-telemetry-chart
                  [data]="getHistoryTrend(prod.device_id)"
                  label="Battery %"
                  strokeColor="#00f0ff"
                  fillColor="rgba(0, 240, 255, 0.1)"
                  [height]="65">
                </app-telemetry-chart>
              </div>

              <!-- Capabilities Chips -->
              <div class="capabilities-section">
                <span class="section-label">CAPABILITIES ({{ prod.capabilities.length }})</span>
                <div class="cap-chips">
                  @for (cap of prod.capabilities; track cap.capability_name) {
                    <span class="cap-chip" [title]="cap.description || cap.capability_name">
                      {{ cap.capability_name }}
                    </span>
                  }
                  @if (prod.capabilities.length === 0) {
                    <span class="no-cap">Standard telemetry report</span>
                  }
                </div>
              </div>

              <!-- Footer Actions -->
              <div class="card-footer">
                <div class="heartbeat-meta">
                  <span class="hb-dot"></span>
                  Last contact: {{ formatTime(prod.last_heartbeat) }}
                </div>
                <a [routerLink]="['/products', prod.device_id]" class="btn btn-primary btn-sm">
                  Inspect & Command
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                  </svg>
                </a>
              </div>
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .products-container {
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

    .stat-chip.online {
      background: rgba(16, 185, 129, 0.1);
      border-color: rgba(16, 185, 129, 0.3);
      color: #10b981;
    }

    .stat-chip .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 6px #10b981;
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

    .products-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.25rem;
    }

    .product-card {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      transition: transform 0.2s, border-color 0.2s;
    }

    .product-card:hover {
      transform: translateY(-2px);
      border-color: rgba(0, 240, 255, 0.3);
    }

    .card-header {
      display: flex;
      align-items: flex-start;
      gap: 0.85rem;
    }

    .persona-icon-box {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
    }

    .persona-icon-box.vision {
      background: rgba(0, 240, 255, 0.1);
      border-color: rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .persona-icon-box.glass {
      background: rgba(168, 85, 247, 0.1);
      border-color: rgba(168, 85, 247, 0.3);
      color: #a855f7;
    }

    .persona-icon-box.drone {
      background: rgba(245, 158, 11, 0.1);
      border-color: rgba(245, 158, 11, 0.3);
      color: #f59e0b;
    }

    .persona-icon-box.rover {
      background: rgba(16, 185, 129, 0.1);
      border-color: rgba(16, 185, 129, 0.3);
      color: #10b981;
    }

    .title-meta {
      flex: 1;
      min-width: 0;
    }

    .id-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.15rem;
    }

    .device-id {
      font-size: 0.75rem;
      color: #64748b;
    }

    .sim-badge {
      font-size: 0.6rem;
      font-family: var(--font-mono);
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.3);
      padding: 0.1rem 0.35rem;
      border-radius: 3px;
      font-weight: 700;
    }

    .product-title {
      font-size: 1.05rem;
      font-weight: 600;
      color: #f8fafc;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metrics-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      padding: 0.5rem;
      gap: 0.25rem;
    }

    .metric-cell {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
    }

    .m-label {
      font-size: 0.6rem;
      color: #64748b;
      font-weight: 700;
      letter-spacing: 0.05em;
    }

    .m-val {
      font-size: 0.8rem;
      font-weight: 600;
      color: #e2e8f0;
      margin-top: 0.1rem;
    }

    .section-label {
      display: block;
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
      margin-bottom: 0.4rem;
    }

    .trend-section {
      background: rgba(0, 0, 0, 0.2);
      border-radius: 6px;
      padding: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.03);
    }

    .capabilities-section {
      display: flex;
      flex-direction: column;
    }

    .cap-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }

    .cap-chip {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #cbd5e1;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
    }

    .no-cap {
      font-size: 0.75rem;
      color: #475569;
      font-style: italic;
    }

    .card-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: auto;
      padding-top: 0.75rem;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .heartbeat-meta {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.7rem;
      color: #64748b;
      font-family: var(--font-mono);
    }

    .hb-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
    }
  `]
})
export class ProductsComponent {
  public selectedType = signal<'ALL' | ProductType>('ALL');

  public filteredProducts = computed(() => {
    const list = this.appState.products();
    const filter = this.selectedType();
    if (filter === 'ALL') return list;
    return list.filter(p => p.product_type === filter);
  });

  constructor(public appState: AppStateService) {}

  public getHistoryTrend(deviceId: string): number[] {
    const history = this.appState.getTelemetryHistory(deviceId);
    if (history.length === 0) {
      return [75, 78, 80, 82, 85, 84, 86, 88];
    }
    return history.map(t => t.battery_level ?? 80);
  }

  public getBatteryColor(pct?: number): string {
    if (pct == null) return '#94a3b8';
    if (pct > 50) return '#10b981';
    if (pct > 20) return '#f59e0b';
    return '#ef4444';
  }

  public formatTime(ts?: number): string {
    if (!ts) return 'Just now';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 5) return 'Just now';
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }
}
