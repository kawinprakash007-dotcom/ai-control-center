import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { of } from 'rxjs';
import { AppStateService } from './app-state.service';
import { AtlasApiService } from '../api/atlas-api.service';
import { WebSocketService } from '../websocket/websocket.service';
import { AtlasEvent } from '../models';

describe('AppStateService', () => {
  let service: AppStateService;
  let apiSpy: any;
  let wsSpy: any;

  beforeEach(() => {
    apiSpy = {
      getHealth: vi.fn().mockReturnValue(of({ status: 'healthy', version: '1.0.0', uptime_seconds: 50, components: {} })),
      getReady: vi.fn().mockReturnValue(of({ status: 'ready', initialized: true, subsystems: {} })),
      getDevices: vi.fn().mockReturnValue(of({ devices: [] })),
      getWorldState: vi.fn().mockReturnValue(of(null)),
      getGoals: vi.fn().mockReturnValue(of({ goals: [] })),
      getTraces: vi.fn().mockReturnValue(of({ traces: [] })),
    };

    wsSpy = {
      connect: vi.fn(),
      disconnect: vi.fn(),
      messages$: of(),
      status$: of('connected'),
    };

    TestBed.configureTestingModule({
      providers: [
        AppStateService,
        provideHttpClient(),
        { provide: AtlasApiService, useValue: apiSpy },
        { provide: WebSocketService, useValue: wsSpy },
      ]
    });

    service = TestBed.inject(AppStateService);
  });

  afterEach(() => {
    service.ngOnDestroy();
  });

  it('should initialize and initialize fallback products when device list is empty', () => {
    expect(service).toBeTruthy();
    expect(service.products().length).toBe(4);
    const types = service.products().map(p => p.product_type);
    expect(types).toContain('VISION');
    expect(types).toContain('GLASS');
    expect(types).toContain('DRONE');
    expect(types).toContain('ROVER');
  });

  it('should compute online products count accurately', () => {
    expect(service.onlineProductsCount()).toBe(4);
  });

  it('should bound events array to maximum of 200 items', () => {
    for (let i = 0; i < 250; i++) {
      const ev: AtlasEvent = {
        event_id: `ev_${i}`,
        source_id: 'Test',
        event_type: 'TELEMETRY',
        priority: 'NORMAL',
        message: `Event ${i}`,
        timestamp: Date.now() / 1000,
      };
      service.addEvent(ev);
    }
    expect(service.events().length).toBe(200);
    expect(service.events()[0].event_id).toBe('ev_249');
  });

  it('should set connectionStatus to connected when health probe passes', () => {
    service.refreshHealthAndReady();
    expect(service.connectionStatus()).toBe('connected');
    expect(service.systemHealth()?.status).toBe('healthy');
  });
});
