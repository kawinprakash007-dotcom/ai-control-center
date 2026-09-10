import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Product, WorldEntity, Situation } from '../../../core/models';

@Component({
  selector: 'app-spatial-canvas',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="canvas-container">
      <div class="canvas-header">
        <div class="header-left">
          <span class="mono text-xs text-accent">SPATIAL RADAR & LOCALIZATION</span>
          <span class="text-xs text-dim">| Center: 37.7749° N, 122.4194° W</span>
        </div>
        <div class="header-right mono text-xs text-secondary">
          <span>SCALE: ±50m LOCAL UTM</span>
        </div>
      </div>

      <div class="svg-wrapper">
        <svg viewBox="0 0 500 320" class="radar-svg">
          <!-- Background Grid Lines -->
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.04)" stroke-width="1" />
            </pattern>
            <radialGradient id="radarSweep" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="rgba(0, 240, 255, 0.08)" />
              <stop offset="100%" stop-color="rgba(0, 240, 255, 0)" />
            </radialGradient>
          </defs>

          <!-- Grid Rect -->
          <rect width="100%" height="100%" fill="url(#grid)" />

          <!-- Center Crosshairs & Range Rings -->
          <circle cx="250" cy="160" r="140" fill="none" stroke="rgba(0, 240, 255, 0.12)" stroke-dasharray="3 3" />
          <circle cx="250" cy="160" r="90" fill="none" stroke="rgba(0, 240, 255, 0.18)" />
          <circle cx="250" cy="160" r="40" fill="none" stroke="rgba(0, 240, 255, 0.25)" />
          <line x1="250" y1="20" x2="250" y2="300" stroke="rgba(255, 255, 255, 0.08)" />
          <line x1="20" y1="160" x2="480" y2="160" stroke="rgba(255, 255, 255, 0.08)" />

          <!-- Radar Pulse -->
          <circle cx="250" cy="160" r="140" fill="url(#radarSweep)" />

          <!-- Situations / Hazards -->
          <g *ngFor="let sit of situations; let i = index">
            <circle [attr.cx]="getSituationX(i)" [attr.cy]="getSituationY(i)" r="18" fill="rgba(255, 51, 102, 0.15)" stroke="#ff3366" stroke-width="1.5" stroke-dasharray="2 2" />
            <circle [attr.cx]="getSituationX(i)" [attr.cy]="getSituationY(i)" r="4" fill="#ff3366" />
            <text [attr.x]="getSituationX(i) + 8" [attr.y]="getSituationY(i) - 8" fill="#ff3366" class="mono text-xs font-semibold">
              ! {{ sit.title || 'SITUATION' }}
            </text>
          </g>

          <!-- Digital Twin World Entities -->
          <g *ngFor="let ent of entities; let i = index" (click)="entitySelected.emit(ent)" class="entity-marker" [class.selected]="selectedEntity?.entity_id === ent.entity_id">
            <polygon [attr.points]="getEntityPoints(ent, i)" [attr.fill]="getEntityColor(ent)" opacity="0.85" />
            <text [attr.x]="getEntityX(ent, i) + 8" [attr.y]="getEntityY(ent, i) + 4" [attr.fill]="getEntityColor(ent)" class="mono text-xs">
              {{ ent.name || ent.entity_id }}
            </text>
          </g>

          <!-- Products Coordinates -->
          <g *ngFor="let p of products" (click)="productSelected.emit(p)" class="product-marker" [class.selected]="selectedProduct?.device_id === p.device_id">
            <circle [attr.cx]="getProductX(p)" [attr.cy]="getProductY(p)" r="8" [attr.fill]="getProductColor(p)" class="marker-base" />
            <circle [attr.cx]="getProductX(p)" [attr.cy]="getProductY(p)" r="14" fill="none" [attr.stroke]="getProductColor(p)" stroke-width="1.5" opacity="0.6" class="pulse-ring" />
            <text [attr.x]="getProductX(p) + 12" [attr.y]="getProductY(p) + 4" [attr.fill]="getProductColor(p)" class="mono text-xs font-medium">
              {{ p.display_name }}
            </text>
            <text *ngIf="p.telemetry?.speed_mps" [attr.x]="getProductX(p) + 12" [attr.y]="getProductY(p) + 16" fill="#94a3b8" class="mono text-xs">
              {{ p.telemetry?.speed_mps }} m/s
            </text>
          </g>
        </svg>
      </div>

      <div class="canvas-footer">
        <div class="legend-item"><span class="dot" style="background:#00f0ff"></span> Drone</div>
        <div class="legend-item"><span class="dot" style="background:#10b981"></span> Rover</div>
        <div class="legend-item"><span class="dot" style="background:#38bdf8"></span> Vision</div>
        <div class="legend-item"><span class="dot" style="background:#a855f7"></span> Glass</div>
        <div class="legend-item"><span class="dot" style="background:#f59e0b"></span> Entity / Twin</div>
        <div class="legend-item"><span class="dot" style="background:#ef4444"></span> Incident</div>
      </div>
    </div>
  `,
  styles: [`
    .canvas-container {
      position: relative;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      width: 100%;
    }
    .canvas-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(0, 0, 0, 0.2);
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .text-accent { color: #00f0ff; }
    .text-dim { color: #64748b; }
    .text-secondary { color: #94a3b8; }
    .svg-wrapper {
      width: 100%;
      height: 280px;
      background: #06090e;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .radar-svg {
      width: 100%;
      height: 100%;
    }
    .product-marker, .entity-marker {
      cursor: pointer;
      transition: transform 0.2s;
    }
    .product-marker:hover .marker-base {
      transform: scale(1.25);
    }
    .product-marker.selected .pulse-ring {
      stroke-width: 2.5;
      opacity: 1;
    }
    .entity-marker:hover {
      filter: brightness(1.3);
    }
    .canvas-footer {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding: 8px 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(0, 0, 0, 0.2);
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      color: #94a3b8;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
  `]
})
export class SpatialCanvasComponent {
  @Input() products: Product[] = [];
  @Input() situations: Situation[] = [];
  @Input() entities: WorldEntity[] = [];
  @Input() selectedProduct: Product | null = null;
  @Input() selectedEntity: WorldEntity | null = null;
  @Output() productSelected = new EventEmitter<Product>();
  @Output() entitySelected = new EventEmitter<WorldEntity>();

  getProductX(p: Product): number {
    const type = p.product_type || 'VISION';
    switch (type) {
      case 'VISION': return 120;
      case 'GLASS': return 190;
      case 'DRONE': return 330;
      case 'ROVER': return 370;
      default: return 250;
    }
  }

  getProductY(p: Product): number {
    const type = p.product_type || 'VISION';
    switch (type) {
      case 'VISION': return 100;
      case 'GLASS': return 210;
      case 'DRONE': return 90;
      case 'ROVER': return 220;
      default: return 160;
    }
  }

  getProductColor(p: Product): string {
    const type = p.product_type || 'VISION';
    switch (type) {
      case 'VISION': return '#38bdf8';
      case 'GLASS': return '#a855f7';
      case 'DRONE': return '#00f0ff';
      case 'ROVER': return '#10b981';
      default: return '#94a3b8';
    }
  }

  getSituationX(idx: number): number {
    return 220 + (idx * 60) % 180;
  }

  getSituationY(idx: number): number {
    return 130 + (idx * 50) % 120;
  }

  getEntityX(ent: WorldEntity, idx: number): number {
    if (ent.location && typeof ent.location.longitude === 'number') {
      return 250 + (ent.location.longitude % 100) * 1.5;
    }
    return 160 + (idx * 75) % 200;
  }

  getEntityY(ent: WorldEntity, idx: number): number {
    if (ent.location && typeof ent.location.latitude === 'number') {
      return 160 - (ent.location.latitude % 100) * 1.5;
    }
    return 80 + (idx * 60) % 140;
  }

  getEntityPoints(ent: WorldEntity, idx: number): string {
    const cx = this.getEntityX(ent, idx);
    const cy = this.getEntityY(ent, idx);
    const r = 6;
    return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
  }

  getEntityColor(ent: WorldEntity): string {
    const t = (ent.entity_type || '').toUpperCase();
    if (t === 'HAZARD') return '#ef4444';
    if (t === 'PERSON') return '#00f0ff';
    if (t === 'VEHICLE') return '#a855f7';
    return '#f59e0b';
  }
}
