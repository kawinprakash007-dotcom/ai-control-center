import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span class="badge" [ngClass]="badgeClass">
      <span class="badge-dot" [class.pulse-dot]="pulse"></span>
      <span>{{ label || status }}</span>
    </span>
  `,
  styles: [`
    :host { display: inline-flex; }
  `]
})
export class StatusBadgeComponent {
  @Input() status: string = 'UNKNOWN';
  @Input() label?: string;
  @Input() pulse: boolean = false;

  get badgeClass(): string {
    const s = (this.status || '').toUpperCase();
    if (['HEALTHY', 'ONLINE', 'COMPLETED', 'READY', 'ALLOWED'].includes(s)) {
      return 'badge-healthy';
    }
    if (['WARNING', 'DEGRADED', 'WAITING_FOR_USER', 'PAUSED', 'ASK_PERMISSION'].includes(s)) {
      return 'badge-warning';
    }
    if (['CRITICAL', 'FAULT', 'FAILED', 'ERROR', 'DENIED'].includes(s)) {
      return 'badge-critical';
    }
    if (['RUNNING', 'IN_PROGRESS', 'INITIALIZING', 'CONNECTING', 'INFO'].includes(s)) {
      return 'badge-info';
    }
    return 'badge-offline';
  }
}
