import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-telemetry-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-box">
      <div class="chart-meta">
        <span class="meta-label mono">{{ label }}</span>
        <span class="meta-val mono" [style.color]="effectiveColor">{{ latestValue }}{{ unit }}</span>
      </div>

      <div class="chart-svg-wrap" [style.height.px]="height">
        <svg viewBox="0 0 200 45" preserveAspectRatio="none" class="chart-svg">
          <defs>
            <linearGradient [id]="gradientId" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" [attr.stop-color]="effectiveColor" stop-opacity="0.3" />
              <stop offset="100%" [attr.stop-color]="effectiveColor" stop-opacity="0.0" />
            </linearGradient>
          </defs>

          <!-- Area Fill -->
          <polygon [attr.points]="areaPoints" [attr.fill]="effectiveFill" />

          <!-- Line Path -->
          <polyline [attr.points]="linePoints" fill="none" [attr.stroke]="effectiveColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </div>

      <div class="chart-stats mono text-xs text-dim">
        <span>MIN: {{ minValue }}{{ unit }}</span>
        <span>MAX: {{ maxValue }}{{ unit }}</span>
      </div>
    </div>
  `,
  styles: [`
    .chart-box {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 6px 10px;
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
    }
    .chart-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 10px;
    }
    .meta-label {
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .meta-val {
      font-weight: 600;
      font-size: 12px;
    }
    .chart-svg-wrap {
      width: 100%;
      height: 45px;
    }
    .chart-svg {
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    .chart-stats {
      display: flex;
      justify-content: space-between;
      font-size: 9px;
      color: #64748b;
      font-family: var(--font-mono);
    }
  `]
})
export class TelemetryChartComponent {
  @Input() label: string = 'Telemetry';
  @Input() values: number[] = [80, 82, 85, 84, 83, 86, 88, 87, 89, 90];
  @Input() data?: number[];
  @Input() unit: string = '%';
  @Input() color: string = '#00d4ff';
  @Input() strokeColor?: string;
  @Input() fillColor?: string;
  @Input() height: number = 45;

  get effectiveValues(): number[] {
    if (this.data && this.data.length > 0) return this.data;
    return this.values;
  }

  get effectiveColor(): string {
    return this.strokeColor || this.color || '#00d4ff';
  }

  get effectiveFill(): string {
    if (this.fillColor) return this.fillColor;
    return `url(#${this.gradientId})`;
  }

  get gradientId(): string {
    return 'grad_' + Math.abs(this.label.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0));
  }

  get latestValue(): number {
    const list = this.effectiveValues;
    if (!list || list.length === 0) return 0;
    return Math.round(list[list.length - 1] * 10) / 10;
  }

  get minValue(): number {
    const list = this.effectiveValues;
    if (!list || list.length === 0) return 0;
    return Math.round(Math.min(...list) * 10) / 10;
  }

  get maxValue(): number {
    const list = this.effectiveValues;
    if (!list || list.length === 0) return 100;
    return Math.round(Math.max(...list) * 10) / 10;
  }

  get linePoints(): string {
    const list = this.effectiveValues;
    if (!list || list.length < 2) return '0,20 200,20';
    const min = Math.min(...list);
    const max = Math.max(...list);
    const range = (max - min) || 1;
    const step = 200 / (list.length - 1);

    return list.map((v, i) => {
      const x = i * step;
      const y = 40 - ((v - min) / range) * 32;
      return `${Math.round(x * 10) / 10},${Math.round(y * 10) / 10}`;
    }).join(' ');
  }

  get areaPoints(): string {
    const pts = this.linePoints;
    return `0,45 ${pts} 200,45`;
  }
}
