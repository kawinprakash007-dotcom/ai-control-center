import { Injectable, OnDestroy, signal, computed } from '@angular/core';
import { Subscription, interval, of } from 'rxjs';
import { catchError, switchMap } from 'rxjs/operators';
import { AtlasApiService } from '../api/atlas-api.service';
import { WebSocketService } from '../websocket/websocket.service';
import {
  SystemHealth,
  SystemReadyState,
  ConnectionStatus,
  Product,
  ProductType,
  ProductTelemetry,
  WorldState,
  Trace,
  AtlasEvent,
  Situation,
  Mission,
} from '../models';

const MAX_EVENTS = 200;
const MAX_TELEMETRY_HISTORY = 120;

@Injectable({
  providedIn: 'root'
})
export class AppStateService implements OnDestroy {
  // Signals for authoritative backend state
  public systemHealth = signal<SystemHealth | null>(null);
  public systemReady = signal<SystemReadyState | null>(null);
  public connectionStatus = signal<ConnectionStatus>('disconnected');
  public products = signal<Product[]>([]);
  public worldState = signal<WorldState | null>(null);
  public goals = signal<any[]>([]);
  public traces = signal<Trace[]>([]);
  public events = signal<AtlasEvent[]>([]);
  public activeSituations = signal<Situation[]>([]);
  public activeMissions = signal<Mission[]>([]);

  public isLoading = signal<boolean>(false);
  public errorMessage = signal<string | null>(null);
  public lastUpdated = signal<number>(Date.now());

  // Bounded telemetry history per device
  private telemetryHistoryMap = new Map<string, ProductTelemetry[]>();

  // Computed views
  public onlineProductsCount = computed(() =>
    this.products().filter(p => p.connectivity_status === 'ONLINE').length
  );
  public criticalSituationsCount = computed(() =>
    this.activeSituations().filter(s => s.severity === 'CRITICAL').length
  );
  public activeGoalsCount = computed(() =>
    this.goals().filter(g => g.status === 'running' || g.status === 'created').length
  );
  public isDemoMode = computed(() =>
    Boolean(this.systemReady()?.demo_mode ?? this.systemHealth()?.demo_mode ?? false)
  );

  private pollSubscription: Subscription | null = null;
  private wsSubscription: Subscription | null = null;
  private pollingIntervalMs = 3500;
  private isDestroyed = false;

  constructor(
    private api: AtlasApiService,
    private ws: WebSocketService
  ) {
    this.initWebSocket();
    this.startPolling();
  }

  private initWebSocket(): void {
    this.ws.connect();

    this.wsSubscription = this.ws.messages$.subscribe(msg => {
      if (this.isDestroyed) return;

      const eventItem: AtlasEvent = {
        event_id: msg.event_id || `ev_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        source_id: msg.metadata?.['source_id'] || 'CognitiveRuntime',
        event_type: msg.event_type || 'COGNITIVE_TURN',
        priority: msg.metadata?.['priority'] || 'NORMAL',
        product_id: msg.metadata?.['device_id'] || msg.metadata?.['product_id'],
        message: msg.stage ? `Cognitive Stage: ${msg.stage}` : 'Cognitive event received',
        timestamp: msg.timestamp || Date.now() / 1000,
        correlation_id: msg.turn_id,
        metadata: msg.metadata,
      };

      this.addEvent(eventItem);
    });
  }

  public addEvent(event: AtlasEvent): void {
    const current = this.events();
    const updated = [event, ...current].slice(0, MAX_EVENTS);
    this.events.set(updated);
  }

  public startPolling(): void {
    this.stopPolling();
    this.refreshAll();

    this.pollSubscription = interval(this.pollingIntervalMs).subscribe(() => {
      if (!this.isDestroyed) {
        this.refreshAll();
      }
    });
  }

  public stopPolling(): void {
    if (this.pollSubscription) {
      this.pollSubscription.unsubscribe();
      this.pollSubscription = null;
    }
  }

  public refreshAll(): void {
    this.refreshHealthAndReady();
    this.refreshDevices();
    this.refreshWorldState();
    this.refreshGoals();
    this.refreshTraces();
  }

  public refreshHealthAndReady(): void {
    this.api.getHealth().pipe(
      catchError(() => of(null))
    ).subscribe(health => {
      if (this.isDestroyed) return;
      this.systemHealth.set(health);
      if (!health) {
        this.connectionStatus.set('disconnected');
      } else {
        this.connectionStatus.set('connected');
      }
    });

    this.api.getReady().pipe(
      catchError(() => of(null))
    ).subscribe(ready => {
      if (this.isDestroyed) return;
      this.systemReady.set(ready);
    });
  }

  public refreshDevices(): void {
    this.api.getDevices().pipe(
      catchError(() => of({ devices: [] }))
    ).subscribe(res => {
      if (this.isDestroyed) return;
      const rawDevices = res.devices || [];

      // Normalize products
      const prods: Product[] = rawDevices.map((d: any) => {
        const pType = (d.product_type || d.device_type || 'VISION').toUpperCase() as ProductType;
        const pRole = (d.product_role || 'HYBRID');
        const pHealth = (d.health_status || 'HEALTHY').toUpperCase();
        const pConn = (d.connectivity_status || 'ONLINE').toUpperCase();

        const telem: ProductTelemetry = {
          timestamp: d.last_heartbeat || Date.now() / 1000,
          battery_level: d.metadata?.battery_pct ?? (pType === 'DRONE' ? 88.0 : pType === 'ROVER' ? 92.0 : 100.0),
          health_status: pHealth as any,
          connectivity: pConn as any,
          temperature_celsius: d.metadata?.temperature ?? 24.5,
          location: d.home_location,
          speed_mps: d.metadata?.speed_mps ?? 0.0,
          heading_degrees: d.metadata?.heading_degrees ?? 0.0,
          metrics: d.metadata || {},
        };

        // Track telemetry history
        this.recordTelemetry(d.device_id, telem);

        return {
          device_id: d.device_id,
          display_name: d.display_name || d.device_id,
          device_type: d.device_type || 'robot',
          product_type: pType,
          product_role: pRole,
          contract_version: d.contract_version || '1.0',
          is_simulation: d.is_simulation ?? true,
          connectivity_status: pConn as any,
          health_status: pHealth as any,
          capabilities: d.capabilities || [],
          home_location: d.home_location,
          current_location: d.home_location,
          telemetry: telem,
          last_heartbeat: d.last_heartbeat,
          registered_at: d.registered_at || Date.now() / 1000,
          metadata: d.metadata || {},
        };
      });

      // Ensure canonical 4 personas exist if none registered yet
      if (prods.length === 0) {
        this.products.set(this.getFallbackProducts());
      } else {
        this.products.set(prods);
      }
      this.lastUpdated.set(Date.now());
    });
  }

  private recordTelemetry(deviceId: string, telem: ProductTelemetry): void {
    const list = this.telemetryHistoryMap.get(deviceId) || [];
    const updated = [...list, telem].slice(-MAX_TELEMETRY_HISTORY);
    this.telemetryHistoryMap.set(deviceId, updated);
  }

  public getTelemetryHistory(deviceId: string): ProductTelemetry[] {
    return this.telemetryHistoryMap.get(deviceId) || [];
  }

  public refreshWorldState(): void {
    this.api.getWorldState().pipe(
      catchError(() => of(null))
    ).subscribe(state => {
      if (this.isDestroyed) return;
      this.worldState.set(state);
    });
  }

  public refreshGoals(): void {
    this.api.getGoals().pipe(
      catchError(() => of({ goals: [] }))
    ).subscribe(res => {
      if (this.isDestroyed) return;
      this.goals.set(res.goals || []);
    });
  }

  public refreshTraces(): void {
    this.api.getTraces(50).pipe(
      catchError(() => of({ traces: [] }))
    ).subscribe(res => {
      if (this.isDestroyed) return;
      this.traces.set(res.traces || []);
    });
  }

  private getFallbackProducts(): Product[] {
    return [
      {
        device_id: 'ATLAS_VISION_01',
        display_name: 'ATLAS Vision Alpha',
        device_type: 'SURVEILLANCE_CAMERA',
        product_type: 'VISION',
        product_role: 'FIXED_STATIONARY',
        contract_version: '1.0',
        is_simulation: true,
        connectivity_status: 'ONLINE',
        health_status: 'HEALTHY',
        capabilities: [
          { capability_name: 'detect_objects', description: 'Visual object detection' },
          { capability_name: 'extract_text', description: 'Optical character recognition' }
        ],
        telemetry: { timestamp: Date.now() / 1000, battery_level: 100.0, health_status: 'HEALTHY', connectivity: 'ONLINE' },
        registered_at: Date.now() / 1000,
      },
      {
        device_id: 'ATLAS_GLASS_01',
        display_name: 'ATLAS Glass Recon',
        device_type: 'SMART_GLASSES',
        product_type: 'GLASS',
        product_role: 'MOBILE_RECON',
        contract_version: '1.0',
        is_simulation: true,
        connectivity_status: 'ONLINE',
        health_status: 'HEALTHY',
        capabilities: [
          { capability_name: 'display_hud', description: 'Overlay tactical HUD notification' },
          { capability_name: 'record_view', description: 'First-person field observation' }
        ],
        telemetry: { timestamp: Date.now() / 1000, battery_level: 78.0, health_status: 'HEALTHY', connectivity: 'ONLINE' },
        registered_at: Date.now() / 1000,
      },
      {
        device_id: 'ATLAS_DRONE_01',
        display_name: 'ATLAS Drone Sentinel',
        device_type: 'QUADCOPTER',
        product_type: 'DRONE',
        product_role: 'AERIAL_SURVEILLANCE',
        contract_version: '1.0',
        is_simulation: true,
        connectivity_status: 'ONLINE',
        health_status: 'HEALTHY',
        capabilities: [
          { capability_name: 'takeoff', description: 'Initiate vertical takeoff' },
          { capability_name: 'land', description: 'Safe vertical landing' },
          { capability_name: 'hover', description: 'Maintain current altitude and hover' },
          { capability_name: 'patrol', description: 'Waypoint perimeter patrol' }
        ],
        telemetry: { timestamp: Date.now() / 1000, battery_level: 86.0, health_status: 'HEALTHY', connectivity: 'ONLINE', speed_mps: 4.2 },
        registered_at: Date.now() / 1000,
      },
      {
        device_id: 'ATLAS_ROVER_01',
        display_name: 'ATLAS Rover Vanguard',
        device_type: 'UGV_ROVER',
        product_type: 'ROVER',
        product_role: 'GROUND_PATROL',
        contract_version: '1.0',
        is_simulation: true,
        connectivity_status: 'ONLINE',
        health_status: 'HEALTHY',
        capabilities: [
          { capability_name: 'drive', description: 'Surface wheel motion' },
          { capability_name: 'stop', description: 'Emergency or stationary braking' },
          { capability_name: 'inspect', description: 'Ground sensor sweep' }
        ],
        telemetry: { timestamp: Date.now() / 1000, battery_level: 94.0, health_status: 'HEALTHY', connectivity: 'ONLINE', speed_mps: 1.5 },
        registered_at: Date.now() / 1000,
      },
    ];
  }

  ngOnDestroy(): void {
    this.isDestroyed = true;
    this.stopPolling();
    if (this.wsSubscription) {
      this.wsSubscription.unsubscribe();
      this.wsSubscription = null;
    }
  }
}
