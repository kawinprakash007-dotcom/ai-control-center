import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { signal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { SimulationComponent } from './simulation.component';
import { AtlasApiService } from '../../core/api/atlas-api.service';
import { AppStateService } from '../../core/state/app-state.service';

describe('SimulationComponent', () => {
  let apiSpy: any;
  let appStateMock: any;

  beforeEach(async () => {
    apiSpy = {
      postObservation: vi.fn().mockReturnValue(of({
        success: true,
        status: 'SUCCESS',
        observation_id: 'obs_sim_scn_01_123',
        correlation_id: 'corr_sim_scn_01_123',
        orchestration_id: 'cycle_123',
        cycle_id: 'cycle_123',
        observations_ingested_count: 1,
        situations_fused_count: 1,
        world_transitions_count: 1,
        events_evaluated_count: 1,
        autonomy_decisions_count: 1,
        goals_created_count: 1,
        tool_results_count: 1,
        duration_seconds: 0.05,
        causal_trace: [
          { stage: 'INGRESS', observation_id: 'obs_sim_scn_01_123' },
          { stage: 'SITUATION', situation_id: 'sit_scn_01_123', category: 'SECURITY' },
          { stage: 'WORLD_STATE', transition_id: 'trans_scn_01_123' },
          { stage: 'GOAL', goal_id: 'goal_scn_01_123' },
          { stage: 'TOOL_EXECUTION', call_id: 'call_scn_01_123', capability: 'device_gateway', action: 'intercept' }
        ],
        situations: [
          { situation_id: 'sit_scn_01_123', category: 'SECURITY', severity: 'HIGH', status: 'ACTIVE', title: 'Perimeter Breach' }
        ],
        world_transitions: [
          { transition_id: 'trans_scn_01_123', entity_id: 'sector_4', property_name: 'perimeter_state', transition_type: 'UPDATE' }
        ],
        goals: [
          { goal_id: 'goal_scn_01_123', title: 'Respond to SECURITY: Perimeter Breach', status: 'running', priority: 'HIGH' }
        ],
        tool_results: [
          { capability: 'device_gateway', action: 'intercept', success: true, call_id: 'call_scn_01_123' }
        ]
      })),
    };

    appStateMock = {
      connectionStatus: signal('connected'),
      worldState: signal({
        state_id: 'ws_001',
        version: 3,
        timestamp: Date.now() / 1000,
        entities: [{ entity_id: 'ent_1', entity_type: 'drone', name: 'Drone 1', status: 'active', properties: {}, confidence: 1, last_updated: 0 }],
        conditions: [],
        relationships: [],
        metadata: {}
      }),
      goals: signal([]),
      refreshWorldState: vi.fn(),
      refreshGoals: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [SimulationComponent],
      providers: [
        provideHttpClient(),
        { provide: AtlasApiService, useValue: apiSpy },
        { provide: AppStateService, useValue: appStateMock },
      ]
    }).compileComponents();
  });

  it('1. should render 10 canonical scenarios', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    const cards = compiled.querySelectorAll('.scenario-card');
    expect(cards.length).toBe(10);
  });

  it('2. should render recommended demo badges on SCN-01, SCN-03, SCN-05', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    const recBadges = compiled.querySelectorAll('.rec-badge');
    expect(recBadges.length).toBe(3);
    const recCardIds = Array.from(compiled.querySelectorAll('.scenario-card.recommended .sc-id'))
      .map(el => el.textContent?.trim());
    expect(recCardIds).toEqual(['SCN-01', 'SCN-03', 'SCN-05']);
  });

  it('3. should invoke canonical simulation injection when clicking SCN-01 quick-run button', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    const injectSpy = vi.spyOn(component, 'injectScenario');
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const heroBtn = compiled.querySelector('#btn-run-scn-01') as HTMLButtonElement;
    expect(heroBtn).not.toBeNull();
    heroBtn.click();

    expect(injectSpy).toHaveBeenCalledTimes(1);
    expect(injectSpy.mock.calls[0][0].scenario_id).toBe('SCN-01');
    expect(apiSpy.postObservation).toHaveBeenCalledTimes(1);
  });

  it('4. should invoke canonical simulation injection when clicking SCN-03 and SCN-05 quick-run buttons', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    const injectSpy = vi.spyOn(component, 'injectScenario');
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const btn03 = compiled.querySelector('#btn-run-scn-03') as HTMLButtonElement;
    const btn05 = compiled.querySelector('#btn-run-scn-05') as HTMLButtonElement;
    expect(btn03).not.toBeNull();
    expect(btn05).not.toBeNull();

    btn03.click();
    expect(injectSpy).toHaveBeenCalledWith(expect.objectContaining({ scenario_id: 'SCN-03' }));

    btn05.click();
    expect(injectSpy).toHaveBeenCalledWith(expect.objectContaining({ scenario_id: 'SCN-05' }));
  });

  it('5. should render live mission response panel on successful injection', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[0]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const responsePanel = compiled.querySelector('#mission-response-panel');
    expect(responsePanel).not.toBeNull();
    expect(responsePanel?.textContent).toContain('ATLAS AUTONOMOUS RESPONSE');
    expect(responsePanel?.textContent).toContain('SCN-01 · Multi-Agent Perimeter Breach');
    expect(responsePanel?.textContent).toContain('LIVE ORCHESTRATION PIPELINE');
  });

  it('6. should render structured error without raw stack traces on failed simulation', () => {
    apiSpy.postObservation.mockReturnValue(throwError(() => ({
      status: 422,
      error: { detail: [{ loc: ['body', 'source_id'], msg: 'Field required' }] }
    })));

    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[0]);
    fixture.detectChanges();

    expect(component.actionSuccess()).toBe(false);
    expect(component.actionStatus()).toContain('SIMULATION FAILED — HTTP 422');
    expect(component.actionStatus()).not.toContain('Traceback (most recent call last)');
  });

  it('7. should render result counts correctly', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[0]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const metricsBar = compiled.querySelector('.metrics-bar');
    expect(metricsBar?.textContent).toContain('INGESTED:');
    expect(metricsBar?.textContent).toContain('1');
    expect(metricsBar?.textContent).toContain('FUSED:');
    expect(metricsBar?.textContent).toContain('GOALS:');
    expect(metricsBar?.textContent).toContain('TOOL RESULTS:');
  });

  it('8. should render goal-created state correctly when goal exists', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[0]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.textContent).toContain('Respond to SECURITY: Perimeter Breach');
    expect(compiled.textContent).toContain('goal_scn_01_123');
    expect(compiled.textContent).toContain('HIGH');
  });

  it('9. should render no-goal state correctly when goals_created_count is 0', () => {
    apiSpy.postObservation.mockReturnValue(of({
      success: true,
      status: 'SUCCESS',
      observation_id: 'obs_sim_scn_02_123',
      correlation_id: 'corr_sim_scn_02_123',
      cycle_id: 'cycle_456',
      observations_ingested_count: 1,
      situations_fused_count: 1,
      world_transitions_count: 1,
      goals_created_count: 0,
      tool_results_count: 0,
      duration_seconds: 0.02,
      goals: []
    }));

    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[1]); // SCN-02
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.textContent).toContain('NO AUTONOMOUS GOAL CREATED');
  });

  it('10. should render actual IDs when present and not fabricate missing ones', () => {
    const fixture = TestBed.createComponent(SimulationComponent);
    const component = fixture.componentInstance;
    fixture.detectChanges();

    component.injectScenario(component.scenarios[0]);
    component.isTraceExpanded.set(true);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.textContent).toContain('obs_sim_scn_01_123');
    expect(compiled.textContent).toContain('corr_sim_scn_01_123');
    expect(compiled.textContent).toContain('cycle_123');
    expect(compiled.textContent).toContain('sit_scn_01_123');
    expect(compiled.textContent).toContain('trans_scn_01_123');
  });

  it('11. should accurately reflect disconnected status without breaking the page', () => {
    appStateMock.connectionStatus.set('disconnected');
    const fixture = TestBed.createComponent(SimulationComponent);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const statusPill = compiled.querySelector('.status-pill');
    expect(statusPill?.textContent).toContain('RUNTIME:');
    expect(statusPill?.textContent).toContain('DISCONNECTED');
  });
});
