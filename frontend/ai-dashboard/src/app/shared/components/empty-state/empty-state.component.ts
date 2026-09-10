import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-empty-state',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="empty-state-card card" [class.loading]="isLoading">
      <div class="icon-wrap">
        <span class="icon">{{ icon }}</span>
      </div>
      <h3 class="title">{{ title }}</h3>
      <p class="description">{{ message }}</p>
      <div *ngIf="actionLabel || actionText" class="action-wrap">
        <button class="btn btn-secondary" (click)="handleAction()">
          {{ actionText || actionLabel }}
        </button>
      </div>
    </div>
  `,
  styles: [`
    .empty-state-card {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 40px 24px;
      min-height: 220px;
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
    }
    .icon-wrap {
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
    }
    .icon {
      font-size: 24px;
    }
    .title {
      font-size: 16px;
      font-weight: 600;
      color: #fff;
      margin-bottom: 8px;
    }
    .description {
      font-size: 13px;
      color: #94a3b8;
      max-width: 360px;
      line-height: 1.5;
    }
    .action-wrap {
      margin-top: 18px;
    }
  `]
})
export class EmptyStateComponent {
  @Input() icon: string = '📡';
  @Input() title: string = 'No Data Available';
  @Input() message: string = 'No records have been received from the backend yet.';
  @Input() actionLabel?: string;
  @Input() actionText?: string;
  @Input() isLoading: boolean = false;
  @Output() actionClicked = new EventEmitter<void>();
  @Output() action = new EventEmitter<void>();

  handleAction(): void {
    this.actionClicked.emit();
    this.action.emit();
  }
}
