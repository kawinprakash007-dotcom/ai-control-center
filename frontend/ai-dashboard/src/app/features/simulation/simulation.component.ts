import { Component, signal, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AtlasApiService } from '../../core/api/atlas-api.service';
import { AppStateService } from '../../core/state/app-state.service';
import {
  ScenarioCatalogItem,
  SimulationResultDetails,
  CausalTraceItem,
  SimulationSituation,
  SimulationTransition,
  SimulationGoal,
  SimulationToolResult,
} from '../../core/models';

export type {
  SimulationResultDetails,
  CausalTraceItem,
  SimulationSituation,
  SimulationTransition,
  SimulationGoal,
  SimulationToolResult,
};


@Component({
  selector: 'app-simulation',
  standalone: true,
  imports: [
    CommonModule,
  ],
  template: `
    <div class="sim-container fade-in">
      <!-- Judge-Facing Header -->
      <div class="mission-header card">
        <div class="header-content">
          <div class="brand-row">
            <span class="badge-mission">ATLAS MISSION SIMULATOR</span>
            <span class="header-divider">/</span>
            <span class="sub-badge">DIGITAL TWIN ENVIRONMENT</span>
          </div>
          <p class="mission-subtext">
            Test how ATLAS perceives incidents, builds situational understanding, coordinates autonomous products, and executes closed-loop responses.
          </p>
        </div>

        <div class="status-pills">
          <div class="status-pill" [class.online]="isRuntimeConnected()">
            <span class="pill-dot"></span>
            <span class="pill-lbl">RUNTIME:</span>
            <span class="pill-val">{{ isRuntimeConnected() ? 'CONNECTED' : 'DISCONNECTED' }}</span>
          </div>
          <div class="status-pill active">
            <span class="pill-dot"></span>
            <span class="pill-lbl">SIMULATION:</span>
            <span class="pill-val">ENABLED</span>
          </div>
          <div class="status-pill">
            <span class="pill-lbl">SCENARIOS:</span>
            <span class="pill-val">10 CANONICAL</span>
          </div>
          <div class="status-pill">
            <span class="pill-lbl">REPLAY:</span>
            <span class="pill-val highlight">DETERMINISTIC / READY</span>
          </div>
        </div>
      </div>

      <!-- Action Feedback Banner -->
      @if (actionStatus()) {
        <div class="status-banner" [class.success]="actionSuccess()" [class.error]="!actionSuccess()" [class.running]="isInjecting()">
          <div class="banner-inner">
            <span class="banner-icon font-mono">
              @if (isInjecting()) { [⟳] } @else if (actionSuccess()) { [✓] } @else { [✗] }
            </span>
            <span class="banner-text">{{ actionStatus() }}</span>
          </div>
        </div>
      }

      <!-- Recommended Demonstration Section -->
      <div class="recommended-section card">
        <div class="rec-badge-row">
          <span class="rec-hero-tag">RECOMMENDED DEMONSTRATION</span>
          <span class="rec-focus-tag">AUTONOMOUS INCIDENT INTERCEPT</span>
        </div>

        <div class="rec-body">
          <div class="rec-narrative-col">
            <div class="rec-meta">
              <span class="sc-id font-mono">SCN-01</span>
              <span class="rec-title">Multi-Agent Perimeter Breach</span>
            </div>
            <p class="rec-narrative">
              An intruder is detected inside a restricted perimeter. ATLAS must correlate the observation, update the world state, create a response objective, and coordinate the appropriate autonomous products.
            </p>
            <div class="rec-autonomy-flow font-mono">
              <span class="flow-step">PERCEIVE</span>
              <span class="flow-arr">→</span>
              <span class="flow-step">FUSE</span>
              <span class="flow-arr">→</span>
              <span class="flow-step">UPDATE WORLD</span>
              <span class="flow-arr">→</span>
              <span class="flow-step">DECIDE GOAL</span>
              <span class="flow-arr">→</span>
              <span class="flow-step">COORDINATE</span>
              <span class="flow-arr">→</span>
              <span class="flow-step">EXECUTE</span>
            </div>
          </div>

          <div class="rec-actions-col">
            <button
              id="btn-run-scn-01"
              class="btn-hero"
              [disabled]="isInjecting()"
              (click)="runScenarioById('SCN-01')">
              @if (isInjecting() && selectedScenario()?.scenario_id === 'SCN-01') {
                <span class="spinner-sm"></span> RUNNING SCN-01...
              } @else {
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                RUN SCN-01
              }
            </button>

            <div class="quick-links">
              <span class="quick-label">SECONDARY DEMOS:</span>
              <div class="quick-btns">
                <button
                  id="btn-run-scn-03"
                  class="btn-quick"
                  [disabled]="isInjecting()"
                  (click)="runScenarioById('SCN-03')">
                  RUN SCN-03 (Thermal SAR)
                </button>
                <button
                  id="btn-run-scn-05"
                  class="btn-quick"
                  [disabled]="isInjecting()"
                  (click)="runScenarioById('SCN-05')">
                  RUN SCN-05 (Sensor Failover)
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Live Mission Response Panel (Visible after Run or during Injection) -->
      @if (lastResult(); as res) {
        <div id="mission-response-panel" class="response-panel card" [class.success]="actionSuccess()" [class.error]="!actionSuccess()">
          <!-- Response Title & Status -->
          <div class="resp-header">
            <div class="resp-title-group">
              <span class="resp-eyebrow">ATLAS AUTONOMOUS RESPONSE</span>
              <h2 class="resp-title">{{ res.scenario_id }} · {{ getScenarioTitle(res.scenario_id) }}</h2>
            </div>
            <div class="resp-meta-group">
              <span class="status-badge" [class.badge-success]="actionSuccess()" [class.badge-danger]="!actionSuccess()">
                {{ res.status }}
              </span>
              <span class="resp-time font-mono">{{ res.timestamp | date:'HH:mm:ss' }}</span>
            </div>
          </div>

          <!-- Orchestration Pipeline Flow Visualizer -->
          <div class="pipeline-section">
            <div class="section-title">LIVE ORCHESTRATION PIPELINE</div>
            <div class="pipeline-stages">
              <div class="pipe-stage" [class.success]="res.observations_ingested_count > 0">
                <span class="stage-num font-mono">01</span>
                <span class="stage-name">OBSERVATION</span>
                <span class="stage-state font-mono">{{ res.observations_ingested_count > 0 ? 'SUCCESS' : 'FAILED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="res.situations_fused_count > 0" [class.skipped]="res.situations_fused_count === 0">
                <span class="stage-num font-mono">02</span>
                <span class="stage-name">SITUATION FUSED</span>
                <span class="stage-state font-mono">{{ res.situations_fused_count > 0 ? 'SUCCESS' : 'SKIPPED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="res.world_transitions_count > 0" [class.skipped]="res.world_transitions_count === 0">
                <span class="stage-num font-mono">03</span>
                <span class="stage-name">WORLD UPDATED</span>
                <span class="stage-state font-mono">{{ res.world_transitions_count > 0 ? 'SUCCESS' : 'SKIPPED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="(res.autonomy_decisions_count ?? 0) > 0 || (res.events_evaluated_count ?? 0) > 0 || res.goals_created_count > 0" [class.skipped]="(res.autonomy_decisions_count ?? 0) === 0 && res.goals_created_count === 0">
                <span class="stage-num font-mono">04</span>
                <span class="stage-name">DECISION</span>
                <span class="stage-state font-mono">{{ ((res.autonomy_decisions_count ?? 0) > 0 || res.goals_created_count > 0) ? 'SUCCESS' : 'SKIPPED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="res.goals_created_count > 0" [class.skipped]="res.goals_created_count === 0">
                <span class="stage-num font-mono">05</span>
                <span class="stage-name">GOAL / MISSION</span>
                <span class="stage-state font-mono">{{ res.goals_created_count > 0 ? 'SUCCESS' : 'SKIPPED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="res.tool_results_count > 0" [class.skipped]="res.tool_results_count === 0">
                <span class="stage-num font-mono">06</span>
                <span class="stage-name">TOOL DISPATCH</span>
                <span class="stage-state font-mono">{{ res.tool_results_count > 0 ? 'SUCCESS' : 'SKIPPED' }}</span>
              </div>
              <div class="pipe-arrow">→</div>

              <div class="pipe-stage" [class.success]="actionSuccess()" [class.failed]="!actionSuccess()">
                <span class="stage-num font-mono">07</span>
                <span class="stage-name">RESULT</span>
                <span class="stage-state font-mono">{{ res.status }}</span>
              </div>
            </div>
          </div>

          <!-- Key Metrics Bar -->
          <div class="metrics-bar font-mono">
            <div class="metric-item">
              <span class="m-lbl">CYCLE ID:</span>
              <span class="m-val highlight">{{ res.cycle_id }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">INGESTED:</span>
              <span class="m-val">{{ res.observations_ingested_count }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">FUSED:</span>
              <span class="m-val">{{ res.situations_fused_count }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">TRANSITIONS:</span>
              <span class="m-val">{{ res.world_transitions_count }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">GOALS:</span>
              <span class="m-val" [class.highlight]="res.goals_created_count > 0">{{ res.goals_created_count }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">TOOL RESULTS:</span>
              <span class="m-val">{{ res.tool_results_count }}</span>
            </div>
            <div class="metric-item">
              <span class="m-lbl">DURATION:</span>
              <span class="m-val">{{ res.duration_seconds | number:'1.3-3' }}s</span>
            </div>
          </div>

          <!-- Three-Column Decision & Execution Grid -->
          <div class="resp-grid">
            <!-- Col 1: What ATLAS Did -->
            <div class="resp-card">
              <div class="card-head">
                <span class="head-title">WHAT ATLAS DID</span>
              </div>
              <div class="actions-checklist font-mono">
                <div class="check-item" [class.done]="res.observations_ingested_count > 0">
                  <span class="check-icon">{{ res.observations_ingested_count > 0 ? '✓' : '○' }}</span>
                  <span class="check-text">Received simulation observation</span>
                </div>
                <div class="check-item" [class.done]="res.situations_fused_count > 0">
                  <span class="check-icon">{{ res.situations_fused_count > 0 ? '✓' : '○' }}</span>
                  <span class="check-text">
                    {{ res.situations_fused_count > 0 ? 'Fused situation evidence (' + res.situations_fused_count + ')' : 'Not reported by runtime' }}
                  </span>
                </div>
                <div class="check-item" [class.done]="res.world_transitions_count > 0">
                  <span class="check-icon">{{ res.world_transitions_count > 0 ? '✓' : '○' }}</span>
                  <span class="check-text">
                    {{ res.world_transitions_count > 0 ? 'Updated world state (' + res.world_transitions_count + ' transitions)' : 'Not reported by runtime' }}
                  </span>
                </div>
                <div class="check-item" [class.done]="res.goals_created_count > 0">
                  <span class="check-icon">{{ res.goals_created_count > 0 ? '✓' : '○' }}</span>
                  <span class="check-text">
                    {{ res.goals_created_count > 0 ? 'Created response goal' : 'No autonomous goal required' }}
                  </span>
                </div>
                <div class="check-item" [class.done]="res.tool_results_count > 0">
                  <span class="check-icon">{{ res.tool_results_count > 0 ? '✓' : '○' }}</span>
                  <span class="check-text">
                    {{ res.tool_results_count > 0 ? 'Produced tool result (' + res.tool_results_count + ')' : 'No tool dispatch required' }}
                  </span>
                </div>
              </div>
            </div>

            <!-- Col 2: Product Coordination View -->
            <div class="resp-card">
              <div class="card-head">
                <span class="head-title">ATLAS PRODUCT COORDINATION</span>
              </div>
              <div class="products-list">
                @for (prod of getInvolvedProducts(res.scenario_id); track prod) {
                  <div class="product-item">
                    <div class="prod-left">
                      <span class="prod-name font-mono">{{ prod }}</span>
                      <span class="prod-role">{{ getProductRole(prod) }}</span>
                    </div>
                    <span class="prod-status font-mono" [class.status-active]="getProductStatus(prod, res) === 'ACTIVE (INGRESS SOURCE)'" [class.status-executed]="getProductStatus(prod, res) === 'EXECUTED'" [class.status-involved]="getProductStatus(prod, res) === 'INVOLVED'">
                      {{ getProductStatus(prod, res) }}
                    </span>
                  </div>
                }
              </div>
            </div>

            <!-- Col 3: Mission & Goal Outcome -->
            <div class="resp-card">
              <div class="card-head">
                <span class="head-title">MISSION OUTCOME</span>
              </div>
              @if (res.goals && res.goals.length > 0) {
                <div class="goal-details font-mono">
                  <div class="g-row">
                    <span class="g-lbl">GOAL:</span>
                    <span class="g-val highlight">{{ res.goals[0].title }}</span>
                  </div>
                  <div class="g-row">
                    <span class="g-lbl">GOAL ID:</span>
                    <span class="g-val">{{ res.goals[0].goal_id }}</span>
                  </div>
                  <div class="g-row">
                    <span class="g-lbl">STATUS:</span>
                    <span class="g-val status-green">{{ res.goals[0].status }}</span>
                  </div>
                  <div class="g-row">
                    <span class="g-lbl">PRIORITY:</span>
                    <span class="g-val">{{ res.goals[0].priority }}</span>
                  </div>
                  <div class="g-row">
                    <span class="g-lbl">TOOL DISPATCH:</span>
                    <span class="g-val">{{ res.tool_results_count > 0 ? res.tool_results_count + ' completed' : 'None' }}</span>
                  </div>
                </div>
              } @else {
                <div class="no-goal-box">
                  <span class="no-goal-title font-mono">NO AUTONOMOUS GOAL CREATED</span>
                  <p class="no-goal-desc">
                    Situation severity or policy threshold did not require an autonomous response goal for this frame.
                  </p>
                </div>
              }
            </div>
          </div>

          <!-- World State Delta Section -->
          <div class="ws-delta-section">
            <div class="section-title">WORLD STATE DELTA</div>
            <div class="ws-delta-grid font-mono">
              <div class="delta-col">
                <span class="delta-tag">BEFORE</span>
                <span class="delta-val">
                  {{ res.before_world_version !== undefined ? 'WorldState v' + res.before_world_version + ' (' + (res.before_entities_count ?? 0) + ' entities, ' + (res.before_conditions_count ?? 0) + ' conditions)' : 'Before state not reported by runtime' }}
                </span>
              </div>
              <div class="delta-arr">→</div>
              <div class="delta-col">
                <span class="delta-tag">AFTER</span>
                <span class="delta-val highlight">
                  {{ res.world_transitions_count > 0 ? '+' + res.world_transitions_count + ' transition(s) applied to WorldState' : 'WorldState unchanged' }}
                </span>
              </div>
            </div>

            @if (res.world_transitions && res.world_transitions.length > 0) {
              <div class="transitions-list font-mono">
                @for (tr of res.world_transitions; track tr.transition_id) {
                  <div class="trans-item">
                    <span class="tr-id">{{ tr.transition_id }}</span>
                    <span class="tr-entity">{{ tr.entity_id }}.{{ tr.property_name }}</span>
                    <span class="tr-type">[{{ tr.transition_type }}]</span>
                  </div>
                }
              </div>
            }
          </div>

          <!-- Expandable Causal Trace Lineage -->
          <div class="trace-section">
            <button class="trace-toggle-btn font-mono" (click)="toggleTraceExpanded()">
              <span>{{ isTraceExpanded() ? '[-] HIDE ATLAS TRACE LINEAGE' : '[+] VIEW ATLAS TRACE LINEAGE' }}</span>
              <span class="trace-count">({{ res.causal_trace?.length ?? 5 }} stages recorded)</span>
            </button>

            @if (isTraceExpanded()) {
              <div class="trace-content font-mono">
                <div class="trace-line">
                  <span class="tr-stage">OBSERVATION</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">{{ res.observation_id }}</span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">CORRELATION</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">{{ res.correlation_id }}</span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">CAUSATION</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">{{ res.causation_id || 'cause_catalog_' + res.scenario_id.toLowerCase().replace('-', '_') }}</span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">CYCLE ID</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">{{ res.cycle_id }}</span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">SITUATION</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">
                    {{ res.situations && res.situations.length > 0 ? res.situations[0].situation_id + ' [' + res.situations[0].category + ']' : (res.situations_fused_count > 0 ? 'Fused (' + res.situations_fused_count + ')' : '[ID not reported]') }}
                  </span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">TRANSITION</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">
                    {{ res.world_transitions && res.world_transitions.length > 0 ? res.world_transitions[0].transition_id : (res.world_transitions_count > 0 ? 'Applied (' + res.world_transitions_count + ')' : '[ID not reported]') }}
                  </span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">GOAL</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">
                    {{ res.goals && res.goals.length > 0 ? res.goals[0].goal_id : (res.goals_created_count > 0 ? 'Goal created' : '[No goal created]') }}
                  </span>
                </div>
                <div class="trace-line">
                  <span class="tr-stage">TOOL CALL</span>
                  <span class="tr-arrow">→</span>
                  <span class="tr-data">
                    {{ res.tool_results && res.tool_results.length > 0 ? (res.tool_results[0].call_id || res.tool_results[0].capability + '.' + res.tool_results[0].action) : (res.tool_results_count > 0 ? 'Dispatched (' + res.tool_results_count + ')' : '[No tool dispatched]') }}
                  </span>
                </div>
              </div>
            }
          </div>
        </div>
      }

      <!-- Scenario Catalog Section Header -->
      <div class="catalog-heading">
        <h2 class="catalog-title">CANONICAL SCENARIO CATALOG</h2>
        <span class="catalog-subtitle">10 Deterministic Multi-Modal Validation Suites</span>
      </div>

      <!-- Scenarios Grid (10 Canonical Scenarios) -->
      <div class="scenarios-grid">
        @for (scn of scenarios; track scn.scenario_id) {
          <div
            class="scenario-card card"
            [class.selected]="selectedScenario()?.scenario_id === scn.scenario_id"
            [class.recommended]="isRecommended(scn.scenario_id)">

            <div class="sc-header">
              <div class="meta-row">
                <span class="sc-id font-mono">{{ scn.scenario_id }}</span>
                @if (isRecommended(scn.scenario_id)) {
                  <span class="rec-badge font-mono">RECOMMENDED DEMO</span>
                }
                <span class="sc-hash font-mono">{{ scn.deterministic_hash }}</span>
              </div>
              <h3 class="sc-title">{{ scn.title }}</h3>
            </div>

            <p class="sc-desc">{{ scn.description }}</p>

            <!-- Involved Products -->
            <div class="sec-row">
              <span class="sec-label">INVOLVED PRODUCTS:</span>
              <div class="chips-row">
                @for (prod of scn.involved_products; track prod) {
                  <span class="node-chip font-mono">{{ prod }}</span>
                }
              </div>
            </div>

            <!-- Modalities -->
            <div class="sec-row">
              <span class="sec-label">TESTED MODALITIES:</span>
              <div class="chips-row">
                @for (mod of scn.modalities_tested; track mod) {
                  <span class="mod-chip font-mono">{{ mod }}</span>
                }
              </div>
            </div>

            <!-- Autonomy Flow -->
            <div class="sec-row">
              <span class="sec-label">AUTONOMY FLOW:</span>
              <span class="flow-preview font-mono">Detect → Fuse → Decide → Respond</span>
            </div>

            <!-- Expected Outcome -->
            <div class="outcome-box">
              <span class="out-label">EXPECTED BEHAVIOR:</span>
              <p class="out-text">{{ scn.expected_outcome }}</p>
            </div>

            <!-- Trigger Action -->
            <div class="card-footer">
              <button
                [id]="'btn-run-' + scn.scenario_id.toLowerCase()"
                class="btn btn-primary btn-sm w-full"
                [disabled]="isInjecting()"
                (click)="injectScenario(scn)">
                @if (isInjecting() && selectedScenario()?.scenario_id === scn.scenario_id) {
                  <span class="spinner-sm"></span> Simulating...
                } @else {
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                  </svg>
                  RUN SCENARIO
                }
              </button>
            </div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .sim-container {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }

    /* Mission Header */
    .mission-header {
      padding: 1.25rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1.25rem;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 30, 0.95) 100%);
      border: 1px solid rgba(0, 240, 255, 0.2);
      border-radius: 8px;
    }

    .brand-row {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 0.35rem;
    }

    .badge-mission {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: #00f0ff;
      text-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
    }

    .header-divider {
      color: #64748b;
      font-weight: 300;
      font-size: 1.1rem;
    }

    .sub-badge {
      font-size: 0.95rem;
      font-weight: 600;
      color: #94a3b8;
      letter-spacing: 0.05em;
    }

    .mission-subtext {
      font-size: 0.85rem;
      color: #94a3b8;
      max-width: 750px;
      line-height: 1.45;
      margin: 0;
    }

    .status-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #94a3b8;
    }

    .status-pill .pill-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #64748b;
    }

    .status-pill.online .pill-dot {
      background: #10b981;
      box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    }

    .status-pill.active {
      border-color: rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .status-pill.active .pill-dot {
      background: #00f0ff;
      box-shadow: 0 0 6px rgba(0, 240, 255, 0.6);
    }

    .status-pill .pill-lbl {
      color: #64748b;
      font-weight: 600;
    }

    .status-pill .pill-val {
      color: #e2e8f0;
      font-weight: 700;
    }

    .status-pill .pill-val.highlight {
      color: #00f0ff;
    }

    /* Feedback Banner */
    .status-banner {
      padding: 0.75rem 1.25rem;
      border-radius: 6px;
      font-size: 0.85rem;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #ef4444;
    }

    .status-banner.success {
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.35);
      color: #10b981;
    }

    .status-banner.running {
      background: rgba(0, 240, 255, 0.1);
      border-color: rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .banner-inner {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .banner-icon {
      font-weight: 700;
    }

    /* Recommended Demonstration Hero Area */
    .recommended-section {
      padding: 1.35rem 1.5rem;
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(2, 132, 199, 0.02) 100%);
      border: 1px solid rgba(0, 240, 255, 0.35);
      box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
      display: flex;
      flex-direction: column;
      gap: 0.9rem;
    }

    .rec-badge-row {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .rec-hero-tag {
      font-size: 0.7rem;
      font-weight: 800;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      background: #00f0ff;
      color: #050b14;
      font-family: var(--font-mono);
      letter-spacing: 0.06em;
    }

    .rec-focus-tag {
      font-size: 0.7rem;
      font-weight: 700;
      color: #94a3b8;
      font-family: var(--font-mono);
      letter-spacing: 0.05em;
    }

    .rec-body {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1.5rem;
    }

    .rec-narrative-col {
      flex: 1;
      min-width: 320px;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .rec-meta {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .rec-meta .sc-id {
      font-size: 0.9rem;
      color: #00f0ff;
      font-weight: 800;
    }

    .rec-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
    }

    .rec-narrative {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.5;
      margin: 0;
      max-width: 800px;
    }

    .rec-autonomy-flow {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.4rem;
      font-size: 0.7rem;
      margin-top: 0.35rem;
    }

    .flow-step {
      background: rgba(0, 240, 255, 0.1);
      border: 1px solid rgba(0, 240, 255, 0.25);
      color: #00f0ff;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
    }

    .flow-arr {
      color: #64748b;
    }

    .rec-actions-col {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      align-items: flex-end;
    }

    .btn-hero {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      background: #00f0ff;
      color: #050b14;
      font-family: var(--font-mono);
      font-weight: 800;
      font-size: 0.95rem;
      padding: 0.75rem 1.75rem;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      box-shadow: 0 0 16px rgba(0, 240, 255, 0.4);
      transition: all 0.2s ease;
    }

    .btn-hero:hover:not(:disabled) {
      background: #38bdf8;
      transform: translateY(-1px);
      box-shadow: 0 0 24px rgba(0, 240, 255, 0.6);
    }

    .btn-hero:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .quick-links {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .quick-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      font-family: var(--font-mono);
    }

    .quick-btns {
      display: flex;
      gap: 0.4rem;
    }

    .btn-quick {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      font-weight: 600;
      padding: 0.25rem 0.6rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-quick:hover:not(:disabled) {
      background: rgba(0, 240, 255, 0.15);
      border-color: rgba(0, 240, 255, 0.35);
      color: #00f0ff;
    }

    /* Live Mission Response Panel */
    .response-panel {
      padding: 1.5rem;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(16, 185, 129, 0.3);
      box-shadow: 0 0 24px rgba(16, 185, 129, 0.08);
    }

    .response-panel.error {
      border-color: rgba(239, 68, 68, 0.3);
      box-shadow: 0 0 24px rgba(239, 68, 68, 0.08);
    }

    .resp-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 0.85rem;
    }

    .resp-eyebrow {
      font-size: 0.7rem;
      font-weight: 800;
      color: #00f0ff;
      letter-spacing: 0.08em;
      font-family: var(--font-mono);
      display: block;
      margin-bottom: 0.2rem;
    }

    .resp-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
      margin: 0;
    }

    .resp-meta-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .status-badge {
      font-size: 0.75rem;
      font-weight: 800;
      padding: 0.25rem 0.65rem;
      border-radius: 4px;
      font-family: var(--font-mono);
      letter-spacing: 0.04em;
    }

    .badge-success {
      background: rgba(16, 185, 129, 0.2);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.5);
    }

    .badge-danger {
      background: rgba(239, 68, 68, 0.2);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.5);
    }

    .resp-time {
      font-size: 0.8rem;
      color: #94a3b8;
    }

    /* Pipeline Section */
    .pipeline-section {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .section-title {
      font-size: 0.7rem;
      font-weight: 800;
      color: #64748b;
      letter-spacing: 0.06em;
      font-family: var(--font-mono);
    }

    .pipeline-stages {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-wrap: wrap;
    }

    .pipe-stage {
      flex: 1;
      min-width: 110px;
      padding: 0.5rem 0.6rem;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }

    .pipe-stage.success {
      background: rgba(16, 185, 129, 0.08);
      border-color: rgba(16, 185, 129, 0.35);
    }

    .pipe-stage.skipped {
      background: rgba(255, 255, 255, 0.02);
      border-color: rgba(255, 255, 255, 0.06);
      opacity: 0.7;
    }

    .pipe-stage.failed {
      background: rgba(239, 68, 68, 0.08);
      border-color: rgba(239, 68, 68, 0.35);
    }

    .stage-num {
      font-size: 0.6rem;
      color: #64748b;
    }

    .pipe-stage.success .stage-num {
      color: #10b981;
    }

    .stage-name {
      font-size: 0.7rem;
      font-weight: 700;
      color: #cbd5e1;
    }

    .stage-state {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
    }

    .pipe-stage.success .stage-state {
      color: #10b981;
    }

    .pipe-stage.failed .stage-state {
      color: #ef4444;
    }

    .pipe-arrow {
      color: #475569;
      font-weight: 700;
    }

    /* Metrics Bar */
    .metrics-bar {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      padding: 0.6rem 1rem;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      font-size: 0.75rem;
    }

    .metric-item {
      display: flex;
      gap: 0.35rem;
    }

    .metric-item .m-lbl {
      color: #64748b;
    }

    .metric-item .m-val {
      color: #e2e8f0;
      font-weight: 700;
    }

    .metric-item .m-val.highlight {
      color: #00f0ff;
    }

    /* Response Three-Column Grid */
    .resp-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
    }

    .resp-card {
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .card-head {
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 0.4rem;
    }

    .head-title {
      font-size: 0.7rem;
      font-weight: 800;
      color: #00f0ff;
      letter-spacing: 0.06em;
      font-family: var(--font-mono);
    }

    /* Checklist */
    .actions-checklist {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      font-size: 0.75rem;
    }

    .check-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #64748b;
    }

    .check-item.done {
      color: #e2e8f0;
    }

    .check-item.done .check-icon {
      color: #10b981;
      font-weight: 800;
    }

    .check-icon {
      color: #475569;
      width: 14px;
    }

    /* Products Coordination List */
    .products-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .product-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.45rem 0.6rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .prod-left {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }

    .prod-name {
      font-size: 0.75rem;
      font-weight: 700;
      color: #fff;
    }

    .prod-role {
      font-size: 0.65rem;
      color: #94a3b8;
      font-family: var(--font-mono);
    }

    .prod-status {
      font-size: 0.65rem;
      font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
    }

    .status-active {
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .status-executed {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #10b981;
    }

    .status-involved {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #94a3b8;
    }

    /* Goal Details */
    .goal-details {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      font-size: 0.75rem;
    }

    .g-row {
      display: flex;
      gap: 0.5rem;
    }

    .g-lbl {
      color: #64748b;
      min-width: 90px;
    }

    .g-val {
      color: #cbd5e1;
    }

    .g-val.highlight {
      color: #00f0ff;
    }

    .g-val.status-green {
      color: #10b981;
      font-weight: 700;
    }

    .no-goal-box {
      padding: 0.75rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.05);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .no-goal-title {
      font-size: 0.7rem;
      font-weight: 700;
      color: #94a3b8;
    }

    .no-goal-desc {
      font-size: 0.75rem;
      color: #64748b;
      margin: 0;
      line-height: 1.4;
    }

    /* World State Delta */
    .ws-delta-section {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 1rem;
    }

    .ws-delta-grid {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
      font-size: 0.75rem;
    }

    .delta-col {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .delta-tag {
      font-size: 0.65rem;
      font-weight: 800;
      color: #64748b;
    }

    .delta-val {
      color: #cbd5e1;
    }

    .delta-val.highlight {
      color: #10b981;
      font-weight: 700;
    }

    .delta-arr {
      color: #475569;
      font-weight: 700;
    }

    .transitions-list {
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
      font-size: 0.7rem;
      margin-top: 0.4rem;
      border-top: 1px solid rgba(255, 255, 255, 0.04);
      padding-top: 0.5rem;
    }

    .trans-item {
      display: flex;
      gap: 0.6rem;
    }

    .tr-id {
      color: #00f0ff;
    }

    .tr-entity {
      color: #e2e8f0;
    }

    .tr-type {
      color: #10b981;
    }

    /* Trace Section */
    .trace-section {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .trace-toggle-btn {
      background: transparent;
      border: none;
      color: #00f0ff;
      font-size: 0.75rem;
      font-weight: 700;
      cursor: pointer;
      padding: 0.25rem 0;
      text-align: left;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .trace-toggle-btn:hover {
      text-decoration: underline;
    }

    .trace-count {
      color: #64748b;
      font-weight: 400;
    }

    .trace-content {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      padding: 0.75rem 1rem;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 0.75rem;
    }

    .trace-line {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .tr-stage {
      color: #64748b;
      min-width: 120px;
      font-weight: 700;
    }

    .tr-arrow {
      color: #475569;
    }

    .tr-data {
      color: #cbd5e1;
    }

    /* Catalog Heading */
    .catalog-heading {
      margin-top: 0.5rem;
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .catalog-title {
      font-size: 1.1rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: 0.04em;
      margin: 0;
    }

    .catalog-subtitle {
      font-size: 0.8rem;
      color: #64748b;
      font-family: var(--font-mono);
    }

    /* Scenarios Grid */
    .scenarios-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.25rem;
    }

    .scenario-card {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.08);
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }

    .scenario-card:hover {
      transform: translateY(-2px);
      border-color: rgba(0, 240, 255, 0.35);
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }

    .scenario-card.recommended {
      border-color: rgba(0, 240, 255, 0.2);
    }

    .meta-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    }

    .sc-id {
      font-size: 0.75rem;
      color: #00f0ff;
      font-weight: 700;
    }

    .rec-badge {
      font-size: 0.6rem;
      font-weight: 800;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid rgba(0, 240, 255, 0.3);
      color: #00f0ff;
    }

    .sc-hash {
      font-size: 0.65rem;
      color: #64748b;
    }

    .sc-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #fff;
    }

    .sc-desc {
      font-size: 0.8rem;
      color: #94a3b8;
      line-height: 1.4;
    }

    .sec-row {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .sec-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #64748b;
      letter-spacing: 0.05em;
    }

    .flow-preview {
      font-size: 0.7rem;
      color: #94a3b8;
    }

    .chips-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }

    .node-chip {
      font-size: 0.65rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #cbd5e1;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
    }

    .mod-chip {
      font-size: 0.65rem;
      background: rgba(0, 240, 255, 0.1);
      border: 1px solid rgba(0, 240, 255, 0.2);
      color: #00f0ff;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
    }

    .outcome-box {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.04);
      padding: 0.6rem;
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .out-label {
      font-size: 0.65rem;
      font-weight: 700;
      color: #10b981;
    }

    .out-text {
      font-size: 0.75rem;
      color: #e2e8f0;
      line-height: 1.4;
    }

    .card-footer {
      margin-top: auto;
      padding-top: 0.5rem;
    }

    .spinner-sm {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 0.35rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `]
})
export class SimulationComponent {
  private api = inject(AtlasApiService);
  public appState = inject(AppStateService);

  public selectedScenario = signal<ScenarioCatalogItem | null>(null);
  public isInjecting = signal<boolean>(false);
  public actionStatus = signal<string | null>(null);
  public actionSuccess = signal<boolean>(false);
  public lastResult = signal<SimulationResultDetails | null>(null);
  public isTraceExpanded = signal<boolean>(false);

  public isRuntimeConnected = computed(() => this.appState.connectionStatus() === 'connected');

  // The 10 Canonical Digital-Twin Scenarios from Phase 6.5e
  public scenarios: ScenarioCatalogItem[] = [
    {
      scenario_id: 'SCN-01',
      title: 'Multi-Agent Perimeter Breach',
      description: 'Simultaneous intruder detection along Sector 4 perimeter requiring multi-node fusion and tactical intercept.',
      modalities_tested: ['VISUAL', 'SPATIAL', 'TELEMETRY'],
      involved_products: ['ATLAS_VISION_01', 'ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x8f2a101b4e9c',
      expected_outcome: 'Deterministic cross-modal clustering, Policy verification, automated rover dispatch.'
    },
    {
      scenario_id: 'SCN-02',
      title: 'Dense Urban GPS Denial Navigation',
      description: 'Loss of global GPS signal in urban canyon forcing autonomous dead-reckoning and visual-inertial odometry.',
      modalities_tested: ['SPATIAL', 'TELEMETRY', 'TEMPORAL'],
      involved_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x3c9e472a11bf',
      expected_outcome: 'Sensor failover to local frame, temporal velocity integration without positional drift blowup.'
    },
    {
      scenario_id: 'SCN-03',
      title: 'Thermal Hotspot Search & Rescue',
      description: 'Infrared telemetry anomaly detection indicating thermal signature in low-visibility smoke plume.',
      modalities_tested: ['VISUAL', 'TELEMETRY'],
      involved_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x91d582fa038c',
      expected_outcome: 'High-confidence hazard clustering and operator HUD alerting.'
    },
    {
      scenario_id: 'SCN-04',
      title: 'Aerial-Ground Coordinated Intercept',
      description: 'Drone maintains aerial eye-in-the-sky tracking while vectoring ground rover to physical interception.',
      modalities_tested: ['VISUAL', 'SPATIAL', 'TEMPORAL'],
      involved_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x5b70c94e82df',
      expected_outcome: 'Coordinated rendezvous waypoint calculation with sub-meter spatial accuracy.'
    },
    {
      scenario_id: 'SCN-05',
      title: 'Sensor Degradation & Failover',
      description: 'Primary camera occlusion triggers graceful fallback to LiDAR and ground ultrasonic telemetry.',
      modalities_tested: ['VISUAL', 'SPATIAL', 'TELEMETRY'],
      involved_products: ['ATLAS_VISION_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0xaa183fe79102',
      expected_outcome: 'Modality weight shifts from visual to spatial without dropping situational continuity.'
    },
    {
      scenario_id: 'SCN-06',
      title: 'Operator HUD Occlusion & Tactical Guidance',
      description: 'Field operator smart glass HUD receives synthesized spatial breadcrumbs under smoke occlusion.',
      modalities_tested: ['VISUAL', 'SPATIAL'],
      involved_products: ['ATLAS_GLASS_01', 'ATLAS_VISION_01'],
      deterministic_hash: '0x7e290cf451a9',
      expected_outcome: 'Real-time WebSocket HUD packet broadcast with minimal latency.'
    },
    {
      scenario_id: 'SCN-07',
      title: 'Low-Bandwidth Edge-Cloud Partitioning',
      description: 'Network degradation simulated between central brain and edge nodes, testing cached policy resilience.',
      modalities_tested: ['TELEMETRY', 'TEMPORAL'],
      involved_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x12bb940c33ef',
      expected_outcome: 'Edge autonomy preserved; offline buffer synchronization on connection restore.'
    },
    {
      scenario_id: 'SCN-08',
      title: 'Hazardous Material Containment Patrol',
      description: 'Continuous monitoring of containment vessel with chemical and thermal sensor correlation.',
      modalities_tested: ['TELEMETRY', 'SPATIAL'],
      involved_products: ['ATLAS_ROVER_01', 'ATLAS_VISION_01'],
      deterministic_hash: '0x64cf8119ae08',
      expected_outcome: 'Automatic safety zone perimeter creation in WorldState.'
    },
    {
      scenario_id: 'SCN-09',
      title: 'Dynamic Obstacle Swarm Avoidance',
      description: 'Multiple moving non-cooperative obstacles crossing active flight and ground patrol paths.',
      modalities_tested: ['VISUAL', 'SPATIAL', 'TEMPORAL'],
      involved_products: ['ATLAS_DRONE_01', 'ATLAS_ROVER_01'],
      deterministic_hash: '0x43dae812f901',
      expected_outcome: 'Dynamic obstacle vectoring with zero simulated collision events.'
    },
    {
      scenario_id: 'SCN-10',
      title: 'Extreme Weather Cross-Modal Fusion Stress',
      description: 'Heavy rain, wind turbulence, and lighting shifts testing robust perception convergence.',
      modalities_tested: ['VISUAL', 'SPATIAL', 'TELEMETRY', 'TEMPORAL'],
      involved_products: ['ATLAS_VISION_01', 'ATLAS_DRONE_01', 'ATLAS_ROVER_01', 'ATLAS_GLASS_01'],
      deterministic_hash: '0xd09187ec5412',
      expected_outcome: 'Full-stack cross-modal fusion stability with bounded confidence degradation.'
    }
  ];

  public isRecommended(scenarioId: string): boolean {
    return scenarioId === 'SCN-01' || scenarioId === 'SCN-03' || scenarioId === 'SCN-05';
  }

  public getScenarioById(scenarioId: string): ScenarioCatalogItem | undefined {
    return this.scenarios.find(s => s.scenario_id === scenarioId);
  }

  public getScenarioTitle(scenarioId: string): string {
    const scn = this.getScenarioById(scenarioId);
    return scn?.title || scenarioId;
  }

  public getInvolvedProducts(scenarioId: string): string[] {
    const scn = this.getScenarioById(scenarioId);
    return scn?.involved_products || ['ATLAS_VISION_01', 'ATLAS_DRONE_01', 'ATLAS_ROVER_01'];
  }

  public getProductRole(product: string): string {
    if (product.includes('VISION')) return 'ROLE: OBSERVE';
    if (product.includes('DRONE')) return 'ROLE: TRACK / RESPOND';
    if (product.includes('ROVER')) return 'ROLE: INTERCEPT / RESPOND';
    if (product.includes('GLASS')) return 'ROLE: GUIDE / ALERT';
    return 'ROLE: PARTICIPANT';
  }

  public getProductStatus(product: string, res: SimulationResultDetails | null): string {
    if (!res) return 'INVOLVED';
    const isSource = res.observation_id.toLowerCase().includes(product.toLowerCase().replace('atlas_', '')) ||
      (product === 'ATLAS_VISION_01' && (res.scenario_id === 'SCN-01' || res.scenario_id === 'SCN-05' || res.scenario_id === 'SCN-06' || res.scenario_id === 'SCN-08' || res.scenario_id === 'SCN-10')) ||
      (product === 'ATLAS_DRONE_01' && (res.scenario_id === 'SCN-02' || res.scenario_id === 'SCN-03' || res.scenario_id === 'SCN-04' || res.scenario_id === 'SCN-07' || res.scenario_id === 'SCN-09'));

    if (isSource) return 'ACTIVE (INGRESS SOURCE)';
    if (res.tool_results_count > 0 && (product.includes('ROVER') || product.includes('DRONE'))) {
      return 'EXECUTED';
    }
    return 'INVOLVED';
  }

  public toggleTraceExpanded(): void {
    this.isTraceExpanded.update(v => !v);
  }

  public runScenarioById(scenarioId: string): void {
    const scn = this.getScenarioById(scenarioId);
    if (scn) {
      this.injectScenario(scn);
    }
  }

  public buildCanonicalObservation(scn: ScenarioCatalogItem): any {
    const now = Date.now();
    const cleanId = scn.scenario_id.toLowerCase().replace(/-/g, '_');
    const obsId = `obs_sim_${cleanId}_${now}`;
    const corrId = `corr_sim_${cleanId}_${now}`;
    const causeId = `cause_catalog_${cleanId}`;
    const sourceId = (scn.involved_products && scn.involved_products[0]) || 'ATLAS_VISION_01';

    // Canonical ModalityType resolution matching backend enum
    let canonicalModality = 'IMAGE';
    const primaryTestMod = (scn.modalities_tested && scn.modalities_tested[0]) || 'VISUAL';
    switch (primaryTestMod.toUpperCase()) {
      case 'SPATIAL':
      case 'GPS':
        canonicalModality = 'GPS';
        break;
      case 'TELEMETRY':
      case 'TEMPORAL':
        canonicalModality = 'TELEMETRY';
        break;
      case 'AUDIO':
      case 'AUDIO_EVENT':
        canonicalModality = 'AUDIO_EVENT';
        break;
      case 'VOICE':
      case 'VOICE_TRANSCRIPT':
        canonicalModality = 'VOICE_TRANSCRIPT';
        break;
      case 'EVENT':
        canonicalModality = 'EVENT';
        break;
      case 'VISUAL':
      case 'IMAGE':
      default:
        canonicalModality = 'IMAGE';
        break;
    }

    // Default spatial location
    const location = {
      latitude: 37.7749,
      longitude: -122.4194,
      altitude: sourceId.includes('DRONE') ? 25.0 : (sourceId.includes('ROVER') ? 0.5 : 10.0),
      accuracy: 1.0
    };

    // Rich domain scenario payload
    const payloadData: Record<string, any> = {
      scenario_id: scn.scenario_id,
      title: scn.title,
      deterministic_hash: scn.deterministic_hash,
      active_node: sourceId,
      all_involved_nodes: scn.involved_products,
      expected_outcome: scn.expected_outcome,
    };

    if (scn.scenario_id === 'SCN-01') {
      payloadData['incident'] = 'perimeter_breach';
      payloadData['detections'] = [
        { label: 'person_intruder', confidence: 0.98, bounding_box: [0.15, 0.22, 0.45, 0.65], sector: 4 }
      ];
    } else if (scn.scenario_id === 'SCN-02') {
      payloadData['navigation_state'] = 'DEAD_RECKONING';
      payloadData['gps_lock'] = false;
      payloadData['vio_odometry'] = { vx: 1.2, vy: 0.0, vz: -0.4, drift_m: 0.12 };
    } else if (scn.scenario_id === 'SCN-03') {
      payloadData['thermal_anomaly'] = true;
      payloadData['metrics'] = { heat_signature_c: 37.4, ambient_c: 18.2, classification: 'HUMAN' };
    } else {
      payloadData['status'] = 'simulated_frame_active';
      payloadData['metrics'] = { frame_sequence: 1, signal_strength: 0.95 };
    }

    return {
      observation_id: obsId,
      source_id: sourceId,
      device_id: sourceId,
      source_type: 'SIMULATION_TWIN',
      modality: canonicalModality,
      timestamp: now / 1000,
      confidence: 0.98,
      location: location,
      correlation_id: corrId,
      causation_id: causeId,
      payload: payloadData,
      metadata: {
        scenario_id: scn.scenario_id,
        deterministic_hash: scn.deterministic_hash,
        title: scn.title,
        modalities_tested: scn.modalities_tested,
        involved_products: scn.involved_products,
        injected_by: 'digital_twin_ui'
      }
    };
  }

  public injectScenario(scn: ScenarioCatalogItem): void {
    this.selectedScenario.set(scn);
    this.isInjecting.set(true);
    this.actionStatus.set(`Executing canonical simulation: ${scn.scenario_id} - ${scn.title}...`);
    this.actionSuccess.set(false);

    // Capture before snapshot from current app state
    const currentWs = this.appState.worldState();
    const beforeWorldVersion = currentWs?.version ?? 1;
    const beforeEntitiesCount = currentWs?.entities?.length ?? 0;
    const beforeConditionsCount = currentWs?.conditions?.length ?? 0;

    const canonicalReq = this.buildCanonicalObservation(scn);

    this.api.postObservation(canonicalReq).subscribe({
      next: (res) => {
        this.isInjecting.set(false);
        const hasIngested = res && (res.observations_ingested_count > 0 || res.success);
        if (hasIngested) {
          this.actionSuccess.set(true);
          this.actionStatus.set(
            `Simulated scenario [${scn.scenario_id}] executed successfully through Central Orchestration.`
          );
          this.lastResult.set({
            scenario_id: scn.scenario_id,
            status: res.status || 'SUCCESS',
            observation_id: res.observation_id || canonicalReq.observation_id,
            correlation_id: res.correlation_id || canonicalReq.correlation_id,
            causation_id: canonicalReq.causation_id,
            cycle_id: res.cycle_id || res.orchestration_id || 'N/A',
            observations_ingested_count: res.observations_ingested_count ?? 1,
            situations_fused_count: res.situations_fused_count ?? 0,
            world_transitions_count: res.world_transitions_count ?? 0,
            events_evaluated_count: res.events_evaluated_count ?? 0,
            autonomy_decisions_count: res.autonomy_decisions_count ?? 0,
            goals_created_count: res.goals_created_count ?? 0,
            tool_results_count: res.tool_results_count ?? 0,
            duration_seconds: res.duration_seconds ?? 0.0,
            timestamp: Date.now(),
            causal_trace: res.causal_trace || [],
            situations: res.situations || [],
            world_transitions: res.world_transitions || [],
            goals: res.goals || [],
            tool_results: res.tool_results || [],
            before_world_version: beforeWorldVersion,
            before_entities_count: beforeEntitiesCount,
            before_conditions_count: beforeConditionsCount,
          });
          // Refresh live dashboard stores so changes are immediately visible
          this.appState.refreshWorldState();
          this.appState.refreshGoals();
        } else {
          this.actionSuccess.set(false);
          this.actionStatus.set(
            `Simulation frame rejected by Central Ingress: status=${res?.status || 'UNKNOWN'}`
          );
        }
      },
      error: (err) => {
        this.isInjecting.set(false);
        this.actionSuccess.set(false);
        const detail = err.error?.detail || err.message || 'Check backend status';
        const formattedDetail = typeof detail === 'object' ? JSON.stringify(detail) : detail;
        this.actionStatus.set(`SIMULATION FAILED — HTTP ${err.status || 500}: ${formattedDetail}`);
      }
    });
  }
}
