# ATLAS Phase 6.4: Multi-Product Situation & Mission Intelligence

## 1. Architectural Role & Invariants

ATLAS Phase 6.4 introduces the semantic **Multi-Product Situation & Mission Intelligence** layer across the four ATLAS edge products:
- **ATLAS Vision**: Persistent environmental observation & stationary surveillance
- **ATLAS Glass**: Wearable egocentric perception & Heads-Up Display (HUD) notifications
- **ATLAS Drone**: Aerial verification & rapid intervention
- **ATLAS Rover**: Ground route inspection, spatial navigation & physical intervention

```
+-----------------------------------------------------------------------------------+
|                            ATLAS EDGE PRODUCTS                                     |
|  [ATLAS Vision]       [ATLAS Glass]       [ATLAS Drone]       [ATLAS Rover]       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
                         CentralInputGateway / Sensor Ingress
                                         |
                                         v
                                  SituationFusion
                                         |
                                         v
                           WorldState / Event Autonomy
                                         |
                                         v
          +---------------------------------------------------------------+
          |          PHASE 6.4: MULTI-PRODUCT SITUATION INTELLIGENCE      |
          |  - Cross-Product Situation Clustering                         |
          |  - Spatial (<50m) & Temporal (<300s) Correlation               |
          |  - Source Diversity Corroboration & Confidence Scoring        |
          |  - Semantic Contradiction Detection & Explicit Auditing       |
          |  - Cross-Product Entity Correlation (CANDIDATE/CORRELATED)    |
          +---------------------------------------------------------------+
                                         |
                                         v
          +---------------------------------------------------------------+
          |             PHASE 6.4: MISSION INTELLIGENCE                   |
          |  - ProductRoleSelector (Health, Battery, Semantic Roles)      |
          |  - MissionPlanner (Dependency-Aware Tactical DAGs)            |
          |  - MissionCoordinator (Lifecycle, Step Coordination)         |
          |  - MissionTimeline (Bounded Chronological Trace)              |
          +---------------------------------------------------------------+
                                         |
                                         v
                           AutonomousGoalManager
                                         |
                                         v
                                  CognitiveRuntime
                                         |
                                         v
                                    PolicyEngine
                                         |
                                         v
                                  ToolOrchestrator
                                         |
                                         v
                                   DeviceGateway
                                         |
                                         v
                                 Edge Products / Twins
```

### Critical Architectural Invariants
1. **Single Authority Maintained**: Phase 6.4 does **NOT** introduce a second cognitive brain. It produces semantic situations and tactical mission DAGs which are translated strictly into `Goal` requests submitted to `AutonomousGoalManager`.
2. **Zero Direct GoalStore Mutation**: All goal lifecycle events (`create_goal`, `pause_goal`, `resume_goal`, `cancel_goal`) flow strictly through `AutonomousGoalManager`.
3. **Zero Direct Device / Tool Execution**: Neither situation intelligence nor mission intelligence executes tools or communicates directly with `DeviceGateway`. Dispatch flows down the existing chain: `AutonomousGoalManager -> CognitiveRuntime -> PolicyEngine -> ToolOrchestrator -> DeviceGateway`.
4. **Zero Physical Hardware Drivers**: Strictly pure Python, model-neutral, asynchronous and thread-safe design. No ROS, MAVLink, GPIO, subprocess, or shell dependencies.

---

## 2. Multi-Product Situation Intelligence

The `MultiProductSituationIntelligenceEngine` correlates disparate sensor observations and localized situations into unified, cross-product multi-situations (`MultiProductSituation`).

### Correlation Dimensions
- **Correlation ID Match**: Situations sharing an explicit, non-empty `correlation_id` cluster together deterministically.
- **Entity Correlation**: Situations referencing overlapping entities (`involved_entities`) merge into a shared situation context.
- **Spatial Proximity**: Situations of identical category occurring within 50 meters (Euclidean ground distance approximation) cluster together.
- **Temporal Correlation Window**: Situations occurring more than 300 seconds apart without an explicit correlation ID do not merge.

### Source Diversity Corroboration Boost
Multi-product confirmation provides superior semantic certainty compared to repetitive observations from a single device:
- **Base Confidence**: Weighted average of constituent evidence confidences:
  $$\text{Base} = \frac{\sum c_i^2}{\sum c_i}$$
- **Source Diversity Boost**: $+0.08$ per additional distinct `ProductType` involved, up to a maximum boost of $+0.20$.
- **Same-Source Invariant**: Ten observations from a single camera yield **zero** corroboration boost.
- **Strict Range Bounds**: Final synthesized confidence is strictly clamped to $[0.0, 1.0]$.

### Semantic Contradiction Detection
When products present opposing assessments of reality, ATLAS does **not** discard earlier observations or use arbitrary "last observation wins" overwrites.
- **Detection Pairs**: Automatic semantic contradiction detection between opposing tokens:
  - `detected` $\leftrightarrow$ `not detected` / `no person` / `no target`
  - `present` $\leftrightarrow$ `not present` / `absent`
  - `breach` $\leftrightarrow$ `clear` / `secure`
  - `movement` / `observed` $\leftrightarrow$ `false alarm`
  - `blocked` $\leftrightarrow$ `clear`
- **Audit Preservation**: Contradictions are stored as `SituationContradiction` audit records with both conflicting sources, claims, and timestamps preserved.
- **Confidence Penalty**: An active contradiction applies an immediate deterministic $-0.15$ penalty to overall situation confidence.

---

## 3. Cross-Product Entity Correlation

Entities detected across disparate modalities and viewpoints (e.g. stationary camera overhead, drone aerial view, glass egocentric perspective) are tracked through `EntityCorrelation`.

- **Lifecycle Statuses**:
  - `CANDIDATE`: Entities of matching category detected within a 60-second window.
  - `CORRELATED`: Confirmed temporal and spatial alignment within 15 seconds.
  - `DISPROVED`: Mutually exclusive location, trajectory, or classification disproves correlation.
- **Bounded Capacity**: Strictly bounded retention via `MissionLimits.max_entity_correlations` (default 100).

---

## 4. Product Role Selection & Capability Mapping

The `ProductRoleSelector` maps tactical objectives to edge devices based on role suitability, health, battery state, and semantic capabilities.

### Role Suitability Matrix
| Objective Type | Vision | Glass | Drone | Rover |
| :--- | :---: | :---: | :---: | :---: |
| `VERIFY_INCIDENT` | 0.50 | 0.40 | **0.95** | 0.85 |
| `LOCATE_TARGET` | 0.60 | 0.45 | **0.95** | 0.90 |
| `ASSESS_THREAT` | 0.70 | 0.60 | **0.90** | 0.85 |
| `MAINTAIN_OBSERVATION` | **0.95** | 0.50 | 0.80 | 0.70 |
| `NOTIFY_WEARER` | 0.10 | **0.98** | 0.10 | 0.10 |
| `INSPECT_ROUTE` | 0.50 | 0.40 | 0.85 | **0.95** |
| `DISPATCH_RESPONSE` | 0.20 | 0.40 | **0.90** | **0.90** |
| `MONITOR_AREA` | **0.95** | 0.50 | 0.80 | 0.75 |
| `CONFIRM_RESOLUTION` | **0.90** | 0.75 | 0.85 | 0.80 |

### Device Health & Battery Constraints
- **Offline Devices**: Disconnected devices receive a score of $0.0$ and are excluded from candidate selection.
- **Unhealthy Devices**: Devices in `UNHEALTHY` status receive a score of $0.0$.
- **Degraded Penalty**: Devices in `DEGRADED` status receive a $-0.20$ score penalty.
- **Low Battery Penalty**: Battery levels between 15% and 30% receive a $-0.15$ score penalty.
- **Critical Battery Cutoff**: Battery levels below 15% receive a score of $0.0$ for all travel/mobility objectives (`NOTIFY_WEARER` excepted).

### Semantic Capability Resolution
The selector dynamically resolves high-level objective requirements to specific device actions:
- `flight` / `aerial` $\rightarrow$ `takeoff`, `land`, `hover`, `goto_location`, `navigate_waypoint` (ATLAS Drone)
- `locomotion` / `ground` $\rightarrow$ `move`, `navigate_to`, `patrol_zone`, `dock` (ATLAS Rover)
- `hud` / `wearer_notification` $\rightarrow$ `display_hud`, `send_notification` (ATLAS Glass)
- `camera` / `imaging` $\rightarrow$ optical sensors across Vision, Drone, Rover, and Glass

---

## 5. Mission Planning & Tactical DAGs

The `MissionPlanner` translates multi-situations into structured `Mission` DAGs:
- **Dependency Invariant**: Objectives specify strict prerequisite dependencies (e.g. route inspection and wearer notification depend on successful aerial target verification).
- **Causal Lineage**: Every mission preserves its trigger situation IDs, `correlation_id`, and `causation_id`.
- **Bounded DAG Depth**: Up to `MissionLimits.max_objectives_per_mission` (default 20).

### Standard Mission Templates
1. **Perimeter Security Response** (`SituationCategory.SECURITY`):
   - Obj 1: Aerial verification (`VERIFY_INCIDENT`) $\rightarrow$ Assigned to Drone
   - Obj 2: Ground perimeter route inspection (`INSPECT_ROUTE`, depends on 1) $\rightarrow$ Assigned to Rover
   - Obj 3: Wearer tactical HUD notification (`NOTIFY_WEARER`, depends on 1) $\rightarrow$ Assigned to Glass
   - Obj 4: Continuous stationary optical surveillance (`MAINTAIN_OBSERVATION`) $\rightarrow$ Assigned to Vision
2. **Spatial Anomaly Investigation** (`SituationCategory.ANOMALY`):
   - Obj 1: Target localization (`LOCATE_TARGET`)
   - Obj 2: Resolution confirmation (`CONFIRM_RESOLUTION`, depends on 1)
3. **Hazard Containment & Area Monitoring** (`SituationCategory.ENVIRONMENTAL`):
   - Obj 1: Area containment & stationary monitoring (`MONITOR_AREA`)

---

## 6. Mission Coordination & Evidence Verification

The `MissionCoordinator` manages active missions, tracks objective progression, and coordinates execution with the Central brain.

### Evidence-Driven Completion
Objectives complete **only** when corroborating edge evidence (`ProductEvidence`) arrives meeting defined criteria:
- **Confidence Threshold**: Evidence confidence must exceed `objective.completion_criteria["confidence_threshold"]`.
- **Semantic Keyword Match**: Evidence summary or structured payload must match required keywords (e.g. `TARGET_VERIFIED`, `ROUTE_INSPECTED`, `NOTIFICATION_DISPLAYED`).
- **Dependency Gating**: Dependent objectives remain `PENDING` until all prerequisite objectives achieve `COMPLETED` status.

### Dynamic Replanning & Graceful Degradation
When a product fails during an active mission (e.g. Drone battery depleted or connection lost):
1. The objective is marked for replanning.
2. `MissionPlanner.replan_mission()` evaluates candidate alternate devices.
3. The objective is reassigned to the next best candidate (e.g. Rover inherits ground verification).
4. `Mission.replanning_count` increments deterministically.
5. If `replanning_count` exceeds `MissionLimits.max_replanning_attempts` (default 5), the mission transitions to `FAILED` with explicit diagnostic failure reasons.

---

## 7. Audit Timeline & Replay Support

All mission lifecycle events are logged in the thread-safe `MissionTimeline`:
- Chronological recording of `MISSION_CREATED`, `OBJECTIVE_ACTIVATED`, `EVIDENCE_INGESTED`, `OBJECTIVE_COMPLETED`, `MISSION_PAUSED`, `MISSION_RESUMED`, `MISSION_ABORTED`, and `MISSION_COMPLETED`.
- Bounded retention with FIFO eviction at `max_timeline_entries` (default 200).
- Pure determinism: replay of identical situation histories produces identical mission plans and identical objective execution sequences.
- Credential scrubbing: all evidence payload metadata is automatically scrubbed of sensitive tokens (`api_key`, `password`, `secret`, etc.).

---

## 8. Anti-Patterns Avoided
- **No Secondary Device Gateway**: Mission layer does not talk directly to simulated or physical devices.
- **No Direct GoalStore Modification**: All goal mutations route strictly through `AutonomousGoalManager`.
- **No Hardware-Specific Code**: Pure abstraction layer compatible with any edge adapter or simulated digital twin.
- **No Silent Contradiction Overwriting**: Opposing evidence is explicitly audited and penalized, never silently dropped.
