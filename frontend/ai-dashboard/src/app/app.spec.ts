import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { App } from './app';
import { WebSocketService } from './core/websocket/websocket.service';

describe('App Shell', () => {
  beforeEach(async () => {
    const mockWebSocketService = {
      connect: () => {},
      disconnect: () => {},
      connectionStatus: signal('disconnected'),
      messages$: of(),
    };

    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: WebSocketService, useValue: mockWebSocketService },
      ]
    }).compileComponents();
  });

  it('should create the app shell', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the ATLAS brand title and authoritative badge', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.brand-title')?.textContent).toContain('ATLAS');
    expect(compiled.querySelector('.brand-sub')?.textContent).toContain('CENTRAL COMMAND');
    expect(compiled.querySelector('.arch-badge')?.textContent).toContain('PHASE 6.6');
  });

  it('should contain all 10 operations navigation links', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.navItems.length).toBe(10);
    const routes = app.navItems.map(item => item.route);
    expect(routes).toContain('/command-center');
    expect(routes).toContain('/products');
    expect(routes).toContain('/situations');
    expect(routes).toContain('/missions');
    expect(routes).toContain('/world');
    expect(routes).toContain('/perception');
    expect(routes).toContain('/events');
    expect(routes).toContain('/traces');
    expect(routes).toContain('/simulation');
    expect(routes).toContain('/settings');
  });

  it('should render DEMO MODE badge when demo mode is active', async () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.appState.systemReady.set({
      status: 'ready',
      components: {
        central_orchestrator: true,
        device_gateway: true,
        input_gateway: true,
        fusion_engine: true,
        world_store: true,
        goal_manager: true,
        cognitive_runtime: true,
      },
      simulation_mode: false,
      demo_mode: true,
    });
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.demo-badge')?.textContent).toContain('DEMO MODE');
  });
});
