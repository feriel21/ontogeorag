# Causal chains in the knowledge graph

Causal subgraph: 104 nodes, 80 arcs (causes=49, affects=15, triggers=14, controls=3)
Sources (never caused): 51 · Sinks (never cause): 45
Multi-arc chains found: **30**

A chain is only as reliable as its weakest arc: chain tier = max(arc tiers), chain confidence = min(arc confidences). A three-step chain of Tier-2 arcs is weaker than any single one of them.

## Chains of 3 arcs (13)
- `gas hydrate dissolution -> excess pressure -> formation stress gathering -> developing mtds`  
  relations: causes | causes | causes · weakest tier T1 · min conf 0.4307 ✓all-Tier-1
- `gas hydrate dissolution -> excess pressure -> slope failure -> mass transport deposit`  
  relations: causes | causes | causes · weakest tier T2 · min conf 0.2584
- `rapid rate of sedimentation -> fluid overpressure -> slope failure -> mass transport deposit`  
  relations: causes | triggers | causes · weakest tier T2 · min conf 0.1809
- `rapid sedimentation -> fluid overpressure -> slope failure -> mass transport deposit`  
  relations: causes | triggers | causes · weakest tier T2 · min conf 0.1809
- `steepening up-dip -> unstable condition -> slope failure -> mass transport deposit`  
  relations: causes | causes | causes · weakest tier T2 · min conf 0.1809
- `gas hydrate dissolution -> excess pressure -> slope failure -> incision of lateral ramps`  
  relations: causes | causes | triggers · weakest tier T2 · min conf 0.0
- `gas hydrate dissolution -> excess pressure -> slope failure -> erosional scour`  
  relations: causes | causes | causes · weakest tier T2 · min conf 0.0
- `rapid rate of sedimentation -> fluid overpressure -> slope failure -> incision of lateral ramps`  
  relations: causes | triggers | triggers · weakest tier T2 · min conf 0.0
- `rapid rate of sedimentation -> fluid overpressure -> slope failure -> erosional scour`  
  relations: causes | triggers | causes · weakest tier T2 · min conf 0.0
- `rapid sedimentation -> fluid overpressure -> slope failure -> incision of lateral ramps`  
  relations: causes | triggers | triggers · weakest tier T2 · min conf 0.0
- `rapid sedimentation -> fluid overpressure -> slope failure -> erosional scour`  
  relations: causes | triggers | causes · weakest tier T2 · min conf 0.0
- `steepening up-dip -> unstable condition -> slope failure -> incision of lateral ramps`  
  relations: causes | causes | triggers · weakest tier T2 · min conf 0.0
- `steepening up-dip -> unstable condition -> slope failure -> erosional scour`  
  relations: causes | causes | causes · weakest tier T2 · min conf 0.0

## Chains of 2 arcs (17)
- `earthquake -> slope failure -> mass transport deposit`  
  relations: triggers | causes · weakest tier T1 · min conf 1.0 ✓all-Tier-1
- `water intrusion and fluidisation -> particle segregation -> formation of upper low-density layer`  
  relations: causes | affects · weakest tier T1 · min conf 0.4307 ✓all-Tier-1
- `water intrusion and fluidisation -> particle segregation -> formation of bipartite flow`  
  relations: causes | affects · weakest tier T1 · min conf 0.4307 ✓all-Tier-1
- `water intrusion and fluidisation -> particle segregation -> bipartite flow`  
  relations: causes | causes · weakest tier T1 · min conf 0.4307 ✓all-Tier-1
- `wave-loading effects -> decrease in effective stress -> sediment approaching liquefaction`  
  relations: causes | causes · weakest tier T1 · min conf 0.4307 ✓all-Tier-1
- `earthquake -> slope failure -> incision of lateral ramps`  
  relations: triggers | triggers · weakest tier T1 · min conf 0.0 ✓all-Tier-1
- `excess pore pressure -> slope failure -> mass transport deposit`  
  relations: causes | causes · weakest tier T2 · min conf 0.6
- `seismic loading -> slope failure -> mass transport deposit`  
  relations: triggers | causes · weakest tier T2 · min conf 0.5168
- `methane hydrate dissociation -> high excess pore pressure -> slumping`  
  relations: causes | triggers · weakest tier T2 · min conf 0.1809
- `rapid rate of sedimentation -> fluid overpressure -> slope instability`  
  relations: causes | triggers · weakest tier T2 · min conf 0.1809
- `rapid sedimentation -> fluid overpressure -> slope instability`  
  relations: causes | triggers · weakest tier T2 · min conf 0.1809
- `sea level fall -> high excess pore pressure -> slumping`  
  relations: causes | triggers · weakest tier T2 · min conf 0.1809
- `earthquake -> slope failure -> erosional scour`  
  relations: triggers | causes · weakest tier T2 · min conf 0.0
- `excess pore pressure -> slope failure -> incision of lateral ramps`  
  relations: causes | triggers · weakest tier T2 · min conf 0.0
- `excess pore pressure -> slope failure -> erosional scour`  
  relations: causes | causes · weakest tier T2 · min conf 0.0
- `seismic loading -> slope failure -> incision of lateral ramps`  
  relations: triggers | triggers · weakest tier T2 · min conf 0.0
- `seismic loading -> slope failure -> erosional scour`  
  relations: triggers | causes · weakest tier T2 · min conf 0.0
