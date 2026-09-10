import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AtlasApiService } from '../../core/api/atlas-api.service';
import { AppStateService } from '../../core/state/app-state.service';

@Component({
  selector: 'app-perception',
  standalone: true,
  imports: [
    CommonModule,
  ],
  template: `
    <div class="perception-container fade-in">
      <!-- Header -->
      <div class="header-section">
        <div>
          <h1 class="page-title">Multimodal Perception Console</h1>
          <p class="page-subtitle">Sub-system telemetry, visual bounding detections, spatial point-clouds, and cross-modal temporal fusion</p>
        </div>
        <div class="actions-row">
          <button class="btn btn-primary btn-sm" [disabled]="isIngesting()" (click)="triggerSyntheticObservation()">
            @if (isIngesting()) {
              <span class="spinner-sm"></span> Ingesting...
            } @else {
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
              Ingest Test Observation
            }
          </button>
        </div>
      </div>

      <!-- Ingestion Feedback Hint -->
      @if (ingestionStatus()) {
        <div class="status-banner" [class.success]="ingestionSuccess()">
          {{ ingestionStatus() }}
        </div>
      }

      <!-- Modality Contribution Meters -->
      <div class="card meters-card">
        <span class="card-title">REAL-TIME FUSION MODALITY CONTRIBUTIONS</span>
        <div class="meters-grid">
          <div class="meter-item">
            <div class="m-head">
              <span class="m-name">VISUAL (CAMERAS & OCR)</span>
              <span class="m-pct font-mono">45%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill visual" style="width: 45%"></div></div>
          </div>
          <div class="meter-item">
            <div class="m-head">
              <span class="m-name">SPATIAL (RADAR / DEPTH)</span>
              <span class="m-pct font-mono">30%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill spatial" style="width: 30%"></div></div>
          </div>
          <div class="meter-item">
            <div class="m-head">
              <span class="m-name">TELEMETRY (IMU / SENSORS)</span>
              <span class="m-pct font-mono">15%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill telem" style="width: 15%"></div></div>
          </div>
          <div class="meter-item">
            <div class="m-head">
              <span class="m-name">TEMPORAL TRACKING</span>
              <span class="m-pct font-mono">10%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill temp" style="width: 10%"></div></div>
          </div>
        </div>
      </div>

      <!-- Modality Tabs -->
      <div class="tabs-nav">
        <button class="tab-btn" [class.active]="activeTab() === 'VISUAL'" (click)="activeTab.set('VISUAL')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
          Visual Perception
        </button>
        <button class="tab-btn" [class.active]="activeTab() === 'SPATIAL'" (click)="activeTab.set('SPATIAL')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
          Spatial Depth
        </button>
        <button class="tab-btn" [class.active]="activeTab() === 'TELEMETRY'" (click)="activeTab.set('TELEMETRY')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
          </svg>
          Telemetry Streams
        </button>
        <button class="tab-btn" [class.active]="activeTab() === 'FUSION'" (click)="activeTab.set('FUSION')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="18" cy="5" r="3"></circle>
            <circle cx="6" cy="12" r="3"></circle>
            <circle cx="18" cy="19" r="3"></circle>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
          </svg>
          Cross-Modal Fusion
        </button>
      </div>

      <!-- Tab Content Area -->
      @switch (activeTab()) {
        @case ('VISUAL') {
          <div class="tab-pane card">
            <h3 class="pane-title">VISUAL OBJECT DETECTIONS & BOUNDING BOXES</h3>
            <div class="detections-grid">
              <div class="detection-item">
                <div class="det-header">
                  <span class="label font-mono">DET-V-001 (Intruder Target)</span>
                  <span class="conf font-mono">98.4%</span>
                </div>
                <div class="bbox-preview">
                  <div class="bbox-box" style="top: 25%; left: 35%; width: 25%; height: 45%;">
                    <span class="bbox-tag">PERSON 0.98</span>
                  </div>
                </div>
                <div class="det-meta font-mono">
                  <span>Source: ATLAS_VISION_01</span>
                  <span>Coords: [ymin=0.25, xmin=0.35, ymax=0.70, xmax=0.60]</span>
                </div>
              </div>

              <div class="detection-item">
                <div class="det-header">
                  <span class="label font-mono">DET-V-002 (Hazard Marker)</span>
                  <span class="conf font-mono">92.1%</span>
                </div>
                <div class="bbox-preview">
                  <div class="bbox-box warning" style="top: 60%; left: 15%; width: 20%; height: 30%;">
                    <span class="bbox-tag">FLAMMABLE 0.92</span>
                  </div>
                </div>
                <div class="det-meta font-mono">
                  <span>Source: ATLAS_DRONE_01</span>
                  <span>Coords: [ymin=0.60, xmin=0.15, ymax=0.90, xmax=0.35]</span>
                </div>
              </div>
            </div>
          </div>
        }

        @case ('SPATIAL') {
          <div class="tab-pane card">
            <h3 class="pane-title">SPATIAL RANGING & PROXIMITY BUFFERS</h3>
            <div class="spatial-readouts">
              <div class="s-readout card">
                <span class="s-axis">TARGET RANGE (DRONE TO OBJECT)</span>
                <span class="s-val font-mono">14.24 meters</span>
                <span class="s-sub">Bearing: 042° True North | LiDAR Echo Valid</span>
              </div>
              <div class="s-readout card">
                <span class="s-axis">GROUND ROVER PROXIMITY TO SECTOR PERIMETER</span>
                <span class="s-val font-mono">8.60 meters</span>
                <span class="s-sub">Ultrasonic & Stereo Depth Verified</span>
              </div>
            </div>
          </div>
        }

        @case ('TELEMETRY') {
          <div class="tab-pane card">
            <h3 class="pane-title">HIGH-FREQUENCY SENSOR BUS</h3>
            <table class="data-table">
              <thead>
                <tr>
                  <th>CHANNEL</th>
                  <th>NODE SOURCE</th>
                  <th>SAMPLE VALUE</th>
                  <th>QUALITY</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="font-mono">BAROMETER_PRESSURE</td>
                  <td class="font-mono text-cyan">ATLAS_DRONE_01</td>
                  <td class="font-mono">1013.25 hPa (Rel Alt: 24.5m)</td>
                  <td><span class="quality-tag ok">VALIDATED</span></td>
                </tr>
                <tr>
                  <td class="font-mono">IMU_ACCEL_XYZ</td>
                  <td class="font-mono text-cyan">ATLAS_ROVER_01</td>
                  <td class="font-mono">[0.02, -0.01, 9.81] m/s²</td>
                  <td><span class="quality-tag ok">STABLE</span></td>
                </tr>
                <tr>
                  <td class="font-mono">OPTICAL_FLOW_FPS</td>
                  <td class="font-mono text-cyan">ATLAS_VISION_01</td>
                  <td class="font-mono">29.97 FPS (Latency: 12ms)</td>
                  <td><span class="quality-tag ok">REALTIME</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        }

        @case ('FUSION') {
          <div class="tab-pane card">
            <h3 class="pane-title">CROSS-MODAL CORRELATION CLUSTERS</h3>
            <div class="cluster-list">
              <div class="cluster-card">
                <div class="cl-head">
                  <span class="cl-id font-mono">CLUSTER-FUSION-01</span>
                  <span class="cl-score font-mono">FUSION SCORE: 0.982</span>
                </div>
                <p class="cl-desc">
                  Unified target identity synthesized across 3 independent channels:
                  Camera BoundingBox + Drone Spatial LiDAR + Radio Beacon.
                </p>
                <div class="cl-channels">
                  <span class="badge-chan">Visual: 98%</span>
                  <span class="badge-chan">Spatial: 94%</span>
                  <span class="badge-chan">Telemetry: 91%</span>
                </div>
              </div>
            </div>
          </div>
        }
      }
    </div>
  `,
  styles: [`
    .perception-container {
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

    .status-banner {
      padding: 0.6rem 1rem;
      border-radius: 6px;
      font-size: 0.8rem;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #ef4444;
    }

    .status-banner.success {
      background: rgba(16, 185, 129, 0.15);
      border-color: rgba(16, 185, 129, 0.3);
      color: #10b981;
    }

    .meters-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .card-title {
      font-size: 0.75rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .meters-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .meter-item {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .m-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.7rem;
    }

    .m-name { color: #64748b; font-weight: 600; }
    .m-pct { color: #cbd5e1; }

    .bar-bg {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 3px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
    }

    .bar-fill.visual { background: #00f0ff; }
    .bar-fill.spatial { background: #a855f7; }
    .bar-fill.telem { background: #f59e0b; }
    .bar-fill.temp { background: #10b981; }

    .tabs-nav {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 0.5rem;
      overflow-x: auto;
    }

    .tab-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: transparent;
      border: 1px solid transparent;
      padding: 0.5rem 0.9rem;
      border-radius: 6px;
      font-size: 0.8rem;
      color: #94a3b8;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
    }

    .tab-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }

    .tab-btn.active {
      background: rgba(0, 240, 255, 0.12);
      border-color: rgba(0, 240, 255, 0.35);
      color: #00f0ff;
      font-weight: 600;
    }

    .tab-pane {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .pane-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .detections-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1rem;
    }

    .detection-item {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .det-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    .label { color: #fff; font-weight: 600; }
    .conf { color: #00f0ff; }

    .bbox-preview {
      height: 140px;
      background: #05080f;
      border: 1px dashed rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      position: relative;
    }

    .bbox-box {
      position: absolute;
      border: 2px solid #00f0ff;
      background: rgba(0, 240, 255, 0.15);
      border-radius: 2px;
    }

    .bbox-box.warning {
      border-color: #f59e0b;
      background: rgba(245, 158, 11, 0.15);
    }

    .bbox-tag {
      position: absolute;
      top: -18px;
      left: 0;
      background: #00f0ff;
      color: #000;
      font-size: 0.6rem;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 0 0.3rem;
      border-radius: 2px;
    }

    .bbox-box.warning .bbox-tag {
      background: #f59e0b;
    }

    .det-meta {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      font-size: 0.7rem;
      color: #64748b;
    }

    .spatial-readouts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }

    .s-readout {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .s-axis {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
    }

    .s-val {
      font-size: 1.4rem;
      font-weight: 700;
      color: #a855f7;
    }

    .s-sub {
      font-size: 0.75rem;
      color: #94a3b8;
    }

    .cluster-card {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(0, 240, 255, 0.25);
      border-radius: 6px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .cl-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    .cl-id { color: #00f0ff; font-weight: 700; }
    .cl-score { color: #10b981; font-weight: 700; }

    .cl-desc {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.4;
    }

    .cl-channels {
      display: flex;
      gap: 0.5rem;
    }

    .badge-chan {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.05);
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
      color: #94a3b8;
    }

    .quality-tag {
      font-size: 0.65rem;
      font-family: var(--font-mono);
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
    }

    .quality-tag.ok {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
    }

    .text-cyan { color: #00f0ff; }

    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }

    .data-table th {
      text-align: left;
      padding: 0.5rem;
      color: #64748b;
      font-size: 0.7rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .data-table td {
      padding: 0.5rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
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
export class PerceptionComponent {
  private api = inject(AtlasApiService);
  public appState = inject(AppStateService);

  public activeTab = signal<'VISUAL' | 'SPATIAL' | 'TELEMETRY' | 'FUSION'>('VISUAL');
  public isIngesting = signal<boolean>(false);
  public ingestionStatus = signal<string | null>(null);
  public ingestionSuccess = signal<boolean>(false);

  public triggerSyntheticObservation(): void {
    this.isIngesting.set(true);
    this.ingestionStatus.set(null);

    const testPayload = {
      device_id: 'ATLAS_VISION_01',
      source_type: 'SURVEILLANCE_CAMERA',
      timestamp: Date.now() / 1000,
      modality: 'VISUAL',
      payload: {
        detections: [
          { class_name: 'test_entity', confidence: 0.99, bbox: [0.2, 0.3, 0.4, 0.5] }
        ]
      }
    };

    this.api.postObservation(testPayload).subscribe({
      next: (res) => {
        this.isIngesting.set(false);
        this.ingestionSuccess.set(true);
        this.ingestionStatus.set('Observation successfully ingested into Perception Bus: ' + JSON.stringify(res));
        this.appState.refreshWorldState();
      },
      error: (err) => {
        this.isIngesting.set(false);
        this.ingestionSuccess.set(false);
        this.ingestionStatus.set('Ingestion error: ' + (err.message || 'Check backend endpoint'));
      }
    });
  }
}
