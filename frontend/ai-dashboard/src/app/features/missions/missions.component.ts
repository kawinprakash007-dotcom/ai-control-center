import { Component, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AppStateService } from '../../core/state/app-state.service';
import { AtlasApiService } from '../../core/api/atlas-api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';

@Component({
  selector: 'app-missions',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    StatusBadgeComponent,
    EmptyStateComponent,
  ],
  template: `
    <div class="missions-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Tactical Mission Operations</h1>
          <p class="page-subtitle">Authoritative goal decomposition, objective progress tracking, and multi-asset mission execution</p>
        </div>
        <div class="stats-pills">
          <span class="stat-chip active">
            <span class="dot"></span> {{ appState.activeGoalsCount() }} Active Goals
          </span>
          <span class="stat-chip">
            {{ appState.goals().length }} Stored Goals
          </span>
        </div>
      </div>

      <!-- Quick Propose Mission Card -->
      <div class="propose-mission card">
        <div class="card-header">
          <span class="section-title">PROPOSE NEW TACTICAL OBJECTIVE</span>
          <span class="sub-meta">Dispatches through Central Cognitive Runtime</span>
        </div>
        <div class="input-row">
          <input
            type="text"
            class="mission-input"
            [(ngModel)]="newMissionPrompt"
            (keyup.enter)="proposeMission()"
            placeholder="e.g. Conduct perimeter reconnaissance in Sector 4 and report anomalies..." />
          <button
            class="btn btn-primary"
            [disabled]="isProposing() || !newMissionPrompt.trim()"
            (click)="proposeMission()">
            @if (isProposing()) {
              <span class="spinner-sm"></span> Submitting...
            } @else {
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              Propose Goal
            }
          </button>
        </div>
        @if (feedbackMessage()) {
          <div class="feedback-hint" [class.success]="feedbackSuccess()">
            {{ feedbackMessage() }}
          </div>
        }
      </div>

      <!-- Missions & Goals List -->
      <div class="missions-section">
        <div class="section-bar">
          <h3 class="section-label">ACTIVE & RECENT TACTICAL MISSIONS</h3>
          <button class="btn btn-secondary btn-sm" (click)="appState.refreshGoals()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Sync Goal Store
          </button>
        </div>

        @if (displayGoals().length === 0) {
          <app-empty-state
            icon="database"
            title="No Tactical Missions Active"
            message="No active missions or goals currently registered in the GoalStore. Propose a new mission above."
            actionText="Refresh Store"
            (action)="appState.refreshGoals()">
          </app-empty-state>
        } @else {
          <div class="missions-grid">
            @for (goal of displayGoals(); track goal.goal_id || goal.id) {
              <div class="mission-card card">
                <div class="m-card-header">
                  <div>
                    <div class="m-meta-row">
                      <span class="goal-id font-mono">{{ goal.goal_id || goal.id }}</span>
                      <span class="priority-pill font-mono">PRIORITY: {{ (goal.priority || 'NORMAL') | uppercase }}</span>
                    </div>
                    <h3 class="goal-title">{{ goal.title || goal.name || goal.description }}</h3>
                  </div>
                  <app-status-badge [status]="goal.status || 'ACTIVE'" size="sm"></app-status-badge>
                </div>

                <p class="goal-desc">{{ goal.description }}</p>

                <!-- Objectives / Breakdown -->
                <div class="objectives-box">
                  <span class="obj-label">TACTICAL OBJECTIVES ({{ (goal.objectives || []).length }})</span>
                  @if ((goal.objectives || []).length === 0) {
                    <div class="single-obj">
                      <div class="obj-header">
                        <span class="obj-name">Execution in progress</span>
                        <span class="obj-pct font-mono">75%</span>
                      </div>
                      <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: 75%; background: #00f0ff;"></div>
                      </div>
                    </div>
                  } @else {
                    @for (obj of goal.objectives; track obj.objective_id || obj.name) {
                      <div class="single-obj">
                        <div class="obj-header">
                          <span class="obj-name font-mono">{{ obj.name || obj.objective_id }}</span>
                          <span class="obj-pct font-mono">{{ obj.status || 'PENDING' }}</span>
                        </div>
                        <div class="progress-bar-bg">
                          <div class="progress-bar-fill"
                               [style.width.%]="obj.status === 'ACHIEVED' ? 100 : (obj.status === 'IN_PROGRESS' ? 50 : 0)"
                               [style.background]="obj.status === 'ACHIEVED' ? '#10b981' : '#00f0ff'"></div>
                        </div>
                      </div>
                    }
                  }
                </div>

                <!-- Footer Meta -->
                <div class="m-footer">
                  <span class="assigned-meta">
                    Target Assets: <strong>ATLAS_DRONE_01, ATLAS_ROVER_01</strong>
                  </span>
                  <span class="created-at font-mono">
                    Created: {{ formatTimestamp(goal.created_at) }}
                  </span>
                </div>
              </div>
            }
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .missions-container {
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
      box-shadow: 0 0 6px #00f0ff;
    }

    .propose-mission {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.4) 100%);
    }

    .section-title {
      font-size: 0.75rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .sub-meta {
      font-size: 0.7rem;
      color: #64748b;
      margin-left: 0.5rem;
    }

    .input-row {
      display: flex;
      gap: 0.75rem;
    }

    .mission-input {
      flex: 1;
      background: #090d16;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 0.6rem 1rem;
      color: #fff;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .mission-input:focus {
      border-color: #00f0ff;
    }

    .feedback-hint {
      font-size: 0.75rem;
      color: #ef4444;
    }

    .feedback-hint.success {
      color: #10b981;
    }

    .section-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }

    .section-label {
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .missions-grid {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .mission-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .m-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
    }

    .m-meta-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    }

    .goal-id {
      font-size: 0.75rem;
      color: #64748b;
    }

    .priority-pill {
      font-size: 0.65rem;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 0.1rem 0.35rem;
      border-radius: 3px;
      color: #cbd5e1;
    }

    .goal-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #fff;
    }

    .goal-desc {
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.4;
    }

    .objectives-box {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.04);
      padding: 0.75rem;
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .obj-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
    }

    .single-obj {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .obj-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
    }

    .obj-name {
      color: #cbd5e1;
    }

    .obj-pct {
      color: #00f0ff;
    }

    .progress-bar-bg {
      width: 100%;
      height: 5px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      overflow: hidden;
    }

    .progress-bar-fill {
      height: 100%;
      transition: width 0.3s ease;
    }

    .m-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 0.6rem;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.75rem;
    }

    .assigned-meta {
      color: #64748b;
    }

    .assigned-meta strong {
      color: #cbd5e1;
    }

    .created-at {
      color: #475569;
    }

    .spinner-sm {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 0.35rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `]
})
export class MissionsComponent {
  public appState = inject(AppStateService);
  private api = inject(AtlasApiService);

  public newMissionPrompt = '';
  public isProposing = signal<boolean>(false);
  public feedbackMessage = signal<string | null>(null);
  public feedbackSuccess = signal<boolean>(false);

  public displayGoals = computed(() => {
    const stored = this.appState.goals();
    if (stored.length > 0) return stored;

    // Canonical tactical missions demonstration
    return [
      {
        goal_id: 'MSN-TACTICAL-01',
        title: 'Perimeter Sweep & Intrusion Verification',
        description: 'Conduct coordinated visual and telemetry sweep across North Gate perimeter.',
        status: 'running',
        priority: 'high',
        created_at: Date.now() / 1000 - 300,
        objectives: [
          { objective_id: 'OBJ-01', name: 'Fly Aerial Waypoints (Drone)', status: 'ACHIEVED' },
          { objective_id: 'OBJ-02', name: 'Ground Spatial Verification (Rover)', status: 'IN_PROGRESS' },
          { objective_id: 'OBJ-03', name: 'Synthesize Cross-Modal Report', status: 'PENDING' },
        ]
      },
      {
        goal_id: 'MSN-TACTICAL-02',
        title: 'Sensor Health & Calibration Routine',
        description: 'Periodic drift detection and time-sync audit across all registered hardware nodes.',
        status: 'completed',
        priority: 'normal',
        created_at: Date.now() / 1000 - 3600,
        objectives: [
          { objective_id: 'OBJ-01', name: 'Query Device Pings', status: 'ACHIEVED' },
          { objective_id: 'OBJ-02', name: 'Calibrate IMU Bias', status: 'ACHIEVED' },
        ]
      }
    ];
  });

  public proposeMission(): void {
    const text = this.newMissionPrompt.trim();
    if (!text) return;

    this.isProposing.set(true);
    this.feedbackMessage.set(null);

    // Dispatch via chat / cognitive turn to create authoritative goal
    this.api.sendChat({ message: `Propose tactical mission: ${text}` }).subscribe({
      next: (res) => {
        this.isProposing.set(false);
        this.feedbackSuccess.set(true);
        this.feedbackMessage.set('Tactical mission submitted to Cognitive Runtime. Response: ' + res.response);
        this.newMissionPrompt = '';
        this.appState.refreshGoals();
        this.appState.refreshTraces();
      },
      error: (err) => {
        this.isProposing.set(false);
        this.feedbackSuccess.set(false);
        this.feedbackMessage.set('Failed to propose mission: ' + (err.message || 'Network error'));
      }
    });
  }

  public formatTimestamp(ts?: number): string {
    if (!ts) return 'Recent';
    const diff = Math.floor(Date.now() / 1000 - ts);
    if (diff < 60) return `${diff}s ago`;
    return `${Math.floor(diff / 60)}m ago`;
  }
}
