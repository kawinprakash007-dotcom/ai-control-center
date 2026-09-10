import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-confirmation-dialog',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div *ngIf="isOpen" class="dialog-backdrop" (click)="onBackdropClick($event)">
      <div class="dialog-panel glass-panel">
        <div class="dialog-header">
          <div class="header-title">
            <span class="shield-icon">🛡️</span>
            <h3>{{ title }}</h3>
          </div>
          <button class="close-btn" (click)="handleCancel()">✕</button>
        </div>

        <div class="dialog-body">
          <div class="policy-alert" [ngClass]="isDanger ? 'alert-danger' : 'alert-warning'">
            <div class="alert-icon">{{ isDanger ? '⚠️' : 'ℹ️' }}</div>
            <div class="alert-text">
              <strong>POLICY GATE EVALUATION</strong>
              <p>{{ policyReason || message || 'This action commands physical or simulated actuators and requires confirmation.' }}</p>
            </div>
          </div>

          <div class="action-details card" *ngIf="targetDevice || capability">
            <div class="detail-row" *ngIf="targetDevice">
              <span class="label mono">TARGET DEVICE:</span>
              <span class="val mono text-accent">{{ targetDevice }}</span>
            </div>
            <div class="detail-row" *ngIf="capability">
              <span class="label mono">CAPABILITY:</span>
              <span class="val mono">{{ capability }}</span>
            </div>
            <div class="detail-row" *ngIf="actionName">
              <span class="label mono">ACTION:</span>
              <span class="val mono font-semibold">{{ actionName }}</span>
            </div>
            <div *ngIf="parameters && hasParams" class="detail-row params-row">
              <span class="label mono">PARAMETERS:</span>
              <pre class="params-pre mono">{{ parameters | json }}</pre>
            </div>
          </div>
        </div>

        <div class="dialog-footer">
          <button class="btn btn-secondary" (click)="handleCancel()">
            {{ cancelText || 'Cancel' }}
          </button>
          <button
            class="btn"
            [ngClass]="isDanger ? 'btn-danger' : 'btn-primary'"
            (click)="handleConfirm()"
          >
            {{ confirmText || confirmLabel }}
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .dialog-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(4, 7, 14, 0.82);
      backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      animation: fadeIn 0.15s ease-out;
    }
    .dialog-panel {
      width: 90%;
      max-width: 480px;
      padding: 20px 24px;
      background: #0d1322;
      border: 1px solid rgba(0, 240, 255, 0.3);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
      border-radius: 8px;
    }
    .dialog-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }
    .header-title {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .header-title h3 {
      font-size: 16px;
      font-weight: 600;
      color: #fff;
    }
    .close-btn {
      background: none;
      border: none;
      color: #64748b;
      font-size: 16px;
      cursor: pointer;
      padding: 4px;
    }
    .close-btn:hover { color: #fff; }
    .policy-alert {
      display: flex;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 6px;
      margin-bottom: 14px;
      font-size: 12px;
    }
    .alert-warning {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #f59e0b;
    }
    .alert-danger {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #ef4444;
    }
    .alert-text p {
      margin-top: 2px;
      color: #cbd5e1;
    }
    .action-details {
      padding: 12px 14px;
      margin-bottom: 18px;
      font-size: 12px;
      background: rgba(0, 0, 0, 0.25);
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .detail-row {
      display: flex;
      justify-content: space-between;
      padding: 5px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .detail-row:last-child { border-bottom: none; }
    .label { color: #64748b; }
    .text-accent { color: #00f0ff; }
    .params-row {
      flex-direction: column;
      gap: 6px;
    }
    .params-pre {
      background: #06090e;
      padding: 8px;
      border-radius: 4px;
      font-size: 11px;
      color: #cbd5e1;
      max-height: 90px;
      overflow-y: auto;
    }
    .dialog-footer {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: scale(0.97); }
      to { opacity: 1; transform: scale(1); }
    }
  `]
})
export class ConfirmationDialogComponent {
  @Input() isOpen: boolean = false;
  @Input() title: string = 'Confirm Actuator Command';
  @Input() message?: string;
  @Input() targetDevice: string = '';
  @Input() capability: string = '';
  @Input() actionName: string = '';
  @Input() parameters: any = null;
  @Input() policyReason: string = '';
  @Input() isDanger: boolean = false;
  @Input() confirmLabel: string = 'Authorize & Execute';
  @Input() confirmText?: string;
  @Input() cancelText?: string;

  @Output() confirm = new EventEmitter<void>();
  @Output() confirmed = new EventEmitter<void>();
  @Output() cancel = new EventEmitter<void>();
  @Output() cancelled = new EventEmitter<void>();

  get hasParams(): boolean {
    return this.parameters && Object.keys(this.parameters).length > 0;
  }

  handleConfirm(): void {
    this.confirm.emit();
    this.confirmed.emit();
  }

  handleCancel(): void {
    this.cancel.emit();
    this.cancelled.emit();
  }

  onBackdropClick(event: MouseEvent): void {
    if ((event.target as HTMLElement).classList.contains('dialog-backdrop')) {
      this.handleCancel();
    }
  }
}
