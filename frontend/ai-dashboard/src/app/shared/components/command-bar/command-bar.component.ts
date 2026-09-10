import { Component, NgZone, ChangeDetectorRef, Optional } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AtlasApiService } from '../../../core/api/atlas-api.service';
import { AppStateService } from '../../../core/state/app-state.service';
import { ChatResponse } from '../../../core/models';

@Component({
  selector: 'app-command-bar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="command-bar-container">
      <!-- Response Drawer if available -->
      <div *ngIf="lastResponse || isSubmitting || errorMessage" class="response-drawer glass-panel">
        <div class="drawer-header">
          <div class="header-tag mono text-xs text-accent">
            <span>● COGNITIVE RUNTIME DELIBERATION</span>
            <span *ngIf="lastResponse?.turn_id" class="text-dim">| TURN: {{ lastResponse?.turn_id }}</span>
            <span *ngIf="lastResponse?.execution_time" class="text-dim">| {{ lastResponse?.execution_time | number:'1.2-2' }}s</span>
          </div>
          <button class="drawer-close" (click)="clearResponse()">✕</button>
        </div>

        <div *ngIf="isSubmitting" class="drawer-loading">
          <span class="spinner"></span>
          <span class="mono text-xs text-secondary">Executing cognitive turn via CognitiveRuntime...</span>
        </div>

        <div *ngIf="errorMessage" class="drawer-error mono text-xs text-critical">
          ⚠️ {{ errorMessage }}
        </div>

        <div *ngIf="lastResponse && !isSubmitting" class="drawer-content">
          <div class="query-bubble mono text-xs">
            <span class="role text-dim">OPERATOR:</span> {{ submittedQuery }}
          </div>
          <div class="response-bubble text-sm">
            <span class="role text-accent font-semibold mono text-xs">ATLAS:</span>
            <p>{{ lastResponse.response }}</p>
          </div>
        </div>
      </div>

      <!-- Persistent Command Input Bar -->
      <div class="bar-input-box glass-panel">
        <div class="bar-prefix">
          <span class="pulse-cyan">▶</span>
          <span class="mono text-xs font-semibold text-accent">ASK ATLAS</span>
          <span *ngIf="appState?.isDemoMode()" class="demo-tag mono text-xs">DEMO MODE</span>
        </div>

        <input
          type="text"
          class="command-input"
          [placeholder]="appState?.isDemoMode() ? 'DEMO MODE: open VS Code, launch Chrome, open Notepad, or command fleet...' : 'Command drones, rovers, query world state, or inspect situations...'"
          [(ngModel)]="userMessage"
          (keydown.enter)="submitCommand()"
          [disabled]="isSubmitting"
        />

        <div class="bar-actions">
          <button
            type="button"
            class="mic-btn"
            [class.listening]="isListening"
            (click)="toggleSpeechRecognition()"
            title="Speech input"
          >
            {{ isListening ? '🔴 LISTENING...' : '🎤' }}
          </button>

          <button
            type="button"
            class="btn btn-primary btn-sm"
            [disabled]="isSubmitting || !userMessage.trim()"
            (click)="submitCommand()"
          >
            <span *ngIf="!isSubmitting">EXECUTE ↵</span>
            <span *ngIf="isSubmitting" class="spinner-sm"></span>
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .command-bar-container {
      position: fixed;
      bottom: 16px;
      left: 260px;
      right: 24px;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    @media (max-width: 1024px) {
      .command-bar-container {
        left: 20px;
        right: 20px;
      }
    }
    .response-drawer {
      padding: 14px 18px;
      border: 1px solid var(--border-accent);
      box-shadow: var(--shadow-lg);
      animation: slideUp 0.2s ease-out;
      max-height: 260px;
      overflow-y: auto;
    }
    .drawer-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .drawer-close {
      background: none;
      border: none;
      color: var(--text-dim);
      font-size: 14px;
      cursor: pointer;
    }
    .drawer-loading {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 0;
    }
    .drawer-content {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .query-bubble {
      color: var(--text-dim);
      border-left: 2px solid var(--border-light);
      padding-left: 8px;
    }
    .response-bubble {
      color: var(--text-primary);
      line-height: 1.5;
    }
    .response-bubble p {
      margin-top: 4px;
      white-space: pre-wrap;
    }
    .bar-input-box {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      border: 1px solid var(--border-light);
      box-shadow: var(--shadow-md);
      border-radius: var(--radius-lg);
      background: rgba(14, 20, 34, 0.92);
      backdrop-filter: blur(14px);
    }
    .bar-input-box:focus-within {
      border-color: var(--accent-cyan);
      box-shadow: var(--shadow-glow);
    }
    .bar-prefix {
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .pulse-cyan {
      color: var(--accent-cyan);
      font-size: 10px;
      animation: pulseGlow 1.5s infinite ease-in-out;
    }
    .command-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-size: 14px;
      font-family: var(--font-sans);
    }
    .command-input::placeholder {
      color: var(--text-muted);
      font-size: 13px;
    }
    .bar-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .mic-btn {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      border-radius: var(--radius-sm);
      padding: 5px 9px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .mic-btn:hover {
      border-color: var(--accent-cyan);
      color: var(--text-primary);
    }
    .mic-btn.listening {
      background: var(--status-critical-bg);
      border-color: var(--status-critical);
      color: var(--status-critical);
      animation: pulseGlow 1s infinite;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(0, 212, 255, 0.2);
      border-top-color: var(--accent-cyan);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    .spinner-sm {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 2px solid rgba(0, 0, 0, 0.2);
      border-top-color: #000;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    .demo-tag {
      background: rgba(255, 183, 0, 0.15);
      color: #ffb700;
      border: 1px solid rgba(255, 183, 0, 0.45);
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.65rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      margin-left: 6px;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  `]
})
export class CommandBarComponent {
  userMessage: string = '';
  submittedQuery: string = '';
  lastResponse: ChatResponse | null = null;
  isSubmitting: boolean = false;
  isListening: boolean = false;
  errorMessage: string | null = null;
  private activeTurnId: number = 0;

  constructor(
    private api: AtlasApiService,
    private zone: NgZone,
    private cdr: ChangeDetectorRef,
    @Optional() public appState?: AppStateService
  ) {}

  submitCommand(): void {
    const q = this.userMessage.trim();
    if (!q || this.isSubmitting) return;

    const currentTurn = ++this.activeTurnId;
    let isSettled = false;

    this.submittedQuery = q;
    this.userMessage = '';
    this.isSubmitting = true;
    this.errorMessage = null;

    this.api.postChat(q).subscribe({
      next: (res) => {
        this.zone.run(() => {
          if (isSettled || currentTurn !== this.activeTurnId) return;
          isSettled = true;
          this.lastResponse = res;
          this.isSubmitting = false;
          this.cdr.detectChanges();
        });
      },
      error: (err) => {
        this.zone.run(() => {
          if (isSettled || currentTurn !== this.activeTurnId) return;
          isSettled = true;
          this.isSubmitting = false;
          if (err && err.kind === 'timeout') {
            this.errorMessage = 'ATLAS is still processing the cognitive turn. Please wait a moment or retry.';
          } else if (err && err.kind === 'network') {
            this.errorMessage = 'Connection failure: Backend unreachable or offline.';
          } else if (err && err.kind === 'policy_denial') {
            this.errorMessage = `Policy restriction: ${err.message}`;
          } else {
            this.errorMessage = err.message || 'Cognitive turn failed.';
          }
          this.cdr.detectChanges();
        });
      }
    });
  }

  clearResponse(): void {
    this.activeTurnId++;
    this.lastResponse = null;
    this.errorMessage = null;
  }

  toggleSpeechRecognition(): void {
    if (typeof window === 'undefined') return;
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Speech Recognition is not supported by your browser.');
      return;
    }

    if (this.isListening) {
      this.isListening = false;
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      this.zone.run(() => {
        this.isListening = true;
        this.cdr.detectChanges();
      });
    };

    recognition.onresult = (event: any) => {
      const text = event.results[0][0].transcript;
      this.zone.run(() => {
        this.userMessage = text;
        this.isListening = false;
        this.cdr.detectChanges();
        this.submitCommand();
      });
    };

    recognition.onerror = () => {
      this.zone.run(() => {
        this.isListening = false;
        this.cdr.detectChanges();
      });
    };

    recognition.onend = () => {
      this.zone.run(() => {
        this.isListening = false;
        this.cdr.detectChanges();
      });
    };

    recognition.start();
  }
}
