import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../core/state/app-state.service';
import { AtlasEvent } from '../../core/models';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-events',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    StatusBadgeComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="events-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Live Event Stream & Audit Log</h1>
          <p class="page-subtitle">Real-time WebSocket telemetry stream and authoritative cognitive event sink</p>
        </div>
        <div class="header-actions">
          <button class="btn btn-secondary btn-sm" (click)="exportEventsJson()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Export JSON
          </button>
          <button class="btn btn-secondary btn-sm" (click)="clearEvents()">
            Clear Feed
          </button>
        </div>
      </div>

      <!-- Filters Bar -->
      <div class="filter-bar">
        <div class="search-box">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input
            type="text"
            class="search-input"
            [(ngModel)]="searchQuery"
            placeholder="Search events by keyword, ID, or source..." />
        </div>

        <div class="filter-group">
          <span class="f-label">Priority:</span>
          <button class="filter-btn" [class.active]="priorityFilter() === 'ALL'" (click)="priorityFilter.set('ALL')">All</button>
          <button class="filter-btn" [class.active]="priorityFilter() === 'CRITICAL'" (click)="priorityFilter.set('CRITICAL')">Critical</button>
          <button class="filter-btn" [class.active]="priorityFilter() === 'HIGH'" (click)="priorityFilter.set('HIGH')">High</button>
          <button class="filter-btn" [class.active]="priorityFilter() === 'NORMAL'" (click)="priorityFilter.set('NORMAL')">Normal</button>
        </div>

        <div class="stream-control">
          <label class="toggle-label">
            <input type="checkbox" [(ngModel)]="autoScroll" />
            <span>Auto-Scroll</span>
          </label>
        </div>
      </div>

      <!-- Events Stream List -->
      @if (filteredEvents().length === 0) {
        <app-empty-state
          icon="activity"
          title="No Events Found"
          message="No live event stream matches the active search query or priority filters."
          actionText="Reset Filters"
          (action)="resetFilters()">
        </app-empty-state>
      } @else {
        <div class="events-list">
          @for (ev of filteredEvents(); track ev.event_id) {
            <div class="event-row card" [class.expanded]="expandedId() === ev.event_id" (click)="toggleExpand(ev.event_id)">
              <div class="ev-top">
                <div class="ev-meta">
                  <span class="ev-id font-mono">{{ ev.event_id }}</span>
                  <span class="ev-src font-mono">{{ ev.source_id }}</span>
                  @if (ev.product_id) {
                    <span class="ev-prod font-mono">{{ ev.product_id }}</span>
                  }
                  <span class="ev-type-tag">{{ ev.event_type }}</span>
                </div>
                <div class="ev-right">
                  <app-status-badge [status]="ev.priority || 'NORMAL'" size="sm"></app-status-badge>
                  <span class="ev-time font-mono">{{ formatTime(ev.timestamp) }}</span>
                </div>
              </div>

              <div class="ev-msg">
                {{ ev.message }}
              </div>

              <!-- Expanded JSON Inspector -->
              @if (expandedId() === ev.event_id) {
                <div class="ev-details" (click)="$event.stopPropagation()">
                  <span class="detail-label font-mono">EVENT PAYLOAD & METADATA</span>
                  <pre class="detail-json font-mono">{{ formatJson(ev) }}</pre>
                </div>
              }
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .events-container {
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

    .header-actions {
      display: flex;
      gap: 0.5rem;
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

    .search-box {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(0, 0, 0, 0.3);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      flex: 1;
      min-width: 240px;
    }

    .search-input {
      background: transparent;
      border: none;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
      width: 100%;
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

    .toggle-label {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      color: #94a3b8;
      cursor: pointer;
    }

    .events-list {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .event-row {
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }

    .event-row:hover {
      background: rgba(255, 255, 255, 0.03);
      border-color: rgba(0, 240, 255, 0.2);
    }

    .event-row.expanded {
      border-color: rgba(0, 240, 255, 0.4);
    }

    .ev-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .ev-meta {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .ev-id { font-size: 0.75rem; color: #64748b; }
    .ev-src { font-size: 0.75rem; color: #00f0ff; font-weight: 600; }
    .ev-prod { font-size: 0.7rem; color: #a855f7; }

    .ev-type-tag {
      font-size: 0.65rem;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #cbd5e1;
      padding: 0.1rem 0.35rem;
      border-radius: 3px;
    }

    .ev-right {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .ev-time {
      font-size: 0.75rem;
      color: #64748b;
    }

    .ev-msg {
      font-size: 0.85rem;
      color: #e2e8f0;
      line-height: 1.4;
    }

    .ev-details {
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .detail-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      display: block;
      margin-bottom: 0.35rem;
    }

    .detail-json {
      background: #06090e;
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      color: #94a3b8;
      max-height: 250px;
      overflow-y: auto;
    }
  `]
})
export class EventsComponent {
  public appState = inject(AppStateService);

  public searchQuery = '';
  public priorityFilter = signal<string>('ALL');
  public autoScroll = true;
  public expandedId = signal<string | null>(null);

  public filteredEvents = computed(() => {
    let list = this.appState.events();

    // Default mock events if feed is empty initially
    if (list.length === 0) {
      list = [
        {
          event_id: 'ev-init-001',
          source_id: 'DeviceGateway',
          event_type: 'TELEMETRY',
          priority: 'NORMAL',
          product_id: 'ATLAS_DRONE_01',
          message: 'Authoritative telemetry synchronized: Battery 86%, Altitude 24.5m',
          timestamp: Date.now() / 1000 - 15,
        },
        {
          event_id: 'ev-init-002',
          source_id: 'PolicyEngine',
          event_type: 'POLICY_VERIFICATION',
          priority: 'NORMAL',
          message: 'Pre-flight check passed for command [detect_objects] on ATLAS_VISION_01',
          timestamp: Date.now() / 1000 - 35,
        },
        {
          event_id: 'ev-init-003',
          source_id: 'PerceptionFusion',
          event_type: 'ANOMALY_DETECTION',
          priority: 'HIGH',
          product_id: 'ATLAS_VISION_01',
          message: 'Visual anomaly classified with 0.98 confidence at perimeter',
          timestamp: Date.now() / 1000 - 65,
        }
      ];
    }

    const q = this.searchQuery.toLowerCase().trim();
    if (q) {
      list = list.filter(e =>
        e.event_id.toLowerCase().includes(q) ||
        e.source_id.toLowerCase().includes(q) ||
        (e.message && e.message.toLowerCase().includes(q)) ||
        (e.product_id && e.product_id.toLowerCase().includes(q))
      );
    }

    const pri = this.priorityFilter();
    if (pri !== 'ALL') {
      list = list.filter(e => (e.priority || 'NORMAL').toUpperCase() === pri);
    }

    return list;
  });

  public toggleExpand(id: string): void {
    if (this.expandedId() === id) {
      this.expandedId.set(null);
    } else {
      this.expandedId.set(id);
    }
  }

  public resetFilters(): void {
    this.searchQuery = '';
    this.priorityFilter.set('ALL');
  }

  public clearEvents(): void {
    this.appState.events.set([]);
  }

  public exportEventsJson(): void {
    const data = JSON.stringify(this.appState.events(), null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `atlas_events_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  public formatTime(ts: number): string {
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 5) return 'Just now';
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }

  public formatJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }
}
