# Statement -> triple candidates (to be confirmed)

Fill `confirmed_subject/relation/object` in the CSV. Copy from a candidate when correct; correct it when not; leave blank when no triple applies.

## [PARTIAL_ONLY] #1
*Folds and thrust structures may display an imbricate reflector pattern in seismic data.*

- 1. `fold` --[hasDescriptor]--> `discontinuous` (T1, score 1.15, subj 1.0, obj 0.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [PARTIAL_ONLY] #2
*Burial compaction increases the shear strength of sediments.*

- 1. `wave action` --[affects]--> `equilibrium conditions` (T1, score 0.3, subj 0.3, obj 0.0)
- 2. `lack of support from removed sediment` --[causes]--> `gravitational spreading` (T1, score 0.3, subj 0.3, obj 0.0)
- 3. `methane hydrate dissociation` --[causes]--> `reduction of shear strength of sediment` (T1, score 0.3, subj 0.0, obj 0.3)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #3
*Mass transport deposits can develop within progradational margin clinothemes.*

- 1. `mass transport deposit` --[occursIn]--> `continental margin` (T1, score 1.3, subj 1.0, obj 0.3)
- 2. `debris flow deposit` --[occursIn]--> `non-glaciated margin` (T2, score 0.6, subj 0.3, obj 0.3)
- 3. `debris flow deposit` --[occursIn]--> `passive margin` (T2, score 0.6, subj 0.3, obj 0.3)

## [PARTIAL_ONLY] #4
*Seismic interpretation of MTDs often uses reflector continuity as a descriptor.*

- 1. `mass transport deposit` --[hasDescriptor]--> `low-amplitude` (T1, score 0.75, subj 0.6, obj 0.0)
- 2. `mass transport deposit` --[hasDescriptor]--> `high-amplitude` (T1, score 0.75, subj 0.6, obj 0.0)
- 3. `mass transport deposit` --[hasDescriptor]--> `wedge-shaped` (T1, score 0.75, subj 0.6, obj 0.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [PARTIAL_ONLY] #5
*Layered sediments may produce acoustically laminated seismic reflections.*

- 1. `megaslide` --[hasDescriptor]--> `layered` (T1, score 1.15, subj 0.0, obj 1.0)
- 2. `displaced block` --[hasDescriptor]--> `layered` (T2, score 1.15, subj 0.0, obj 1.0)
- 3. `mass transport deposit` --[hasDescriptor]--> `layered` (T2, score 1.15, subj 0.0, obj 1.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #6
*The basal shear surface of an MTD lies beneath the upper surface of the deposit.*

- 1. `mass transport deposit` --[partOf]--> `basal shear surface` (T1, score 1.6, subj 0.6, obj 1.0)
- 2. `mass transport deposit` --[partOf]--> `upper surface` (T1, score 1.6, subj 0.6, obj 1.0)

## [AMBIGUOUS] #7
*Sea-level lowstands can contribute to slope failure.*

- 1. `sea-level lowstands` --[causes]--> `greatest rates of sedimentation at the outer shelf and upper slope` (T1, score 1.3, subj 1.0, obj 0.3)
- 2. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [AMBIGUOUS] #8
*Rapid sedimentation can trigger the development of fluid overpressure.*

- 1. `rapid rate of sedimentation` --[causes]--> `fluid overpressure` (T1, score 1.3, subj 0.3, obj 1.0)
- 2. `rapid rate of sedimentation` --[triggers]--> `build-up of overpressure` (T1, score 0.75, subj 0.3, obj 0.3)

## [CONFIDENT] #9
*Wave action may destabilize slopes by altering equilibrium conditions.*

- 1. `wave action` --[affects]--> `equilibrium conditions` (T1, score 2.0, subj 1.0, obj 1.0)
- 2. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [CONFIDENT] #10
*High-amplitude blocks can occur within mass transport deposits.*

- 1. `mass transport deposit` --[hasDescriptor]--> `high-amplitude` (T1, score 2.0, subj 1.0, obj 1.0)
- 2. `mass transport deposit` --[hasDescriptor]--> `low-amplitude` (T1, score 1.3, subj 1.0, obj 0.3)
- 3. `remnant and rafted blocks` --[hasDescriptor]--> `high-amplitude` (T1, score 1.3, subj 0.3, obj 1.0)

## [PARTIAL_ONLY] #11
*Mass transport deposits commonly contain internally deformed strata.*

- 1. `growth stratal wedge` --[causes]--> `mass transport deposit` (T1, score 1.15, subj 0.0, obj 1.0)
- 2. `slope failure` --[causes]--> `mass transport deposit` (T1, score 1.15, subj 0.0, obj 1.0)
- 3. `slope failures` --[causes]--> `mass transport deposit` (T1, score 1.15, subj 0.0, obj 1.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #12
*Gas hydrate dissociation may contribute to retrogressive slope failure.*

- 1. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [PARTIAL_ONLY] #13
*Internal reflections within an MTD may be inclined relative to surrounding strata.*

- 1. `channel levee complex` --[hasDescriptor]--> `thin` (T1, score 1.0, subj 0.0, obj 1.0)
- 2. `growth stratal wedge` --[causes]--> `mass transport deposit` (T1, score 0.6, subj 0.0, obj 0.6)
- 3. `slope failure` --[causes]--> `mass transport deposit` (T1, score 0.6, subj 0.0, obj 0.6)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [CONFIDENT] #14
*Mass transport deposits frequently occur on continental slopes.*

- 1. `mass transport deposit` --[occursIn]--> `continental slope` (T2, score 2.15, subj 1.0, obj 1.0)
- 2. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [CONFIDENT] #15
*Mass transport deposits may occur along passive continental margins.*

- 1. `mass transport deposit` --[occursIn]--> `continental margin` (T1, score 2.15, subj 1.0, obj 1.0)
- 2. `debris flow deposit` --[occursIn]--> `non-glaciated margin` (T2, score 0.75, subj 0.3, obj 0.3)
- 3. `debris flow deposit` --[occursIn]--> `passive margin` (T2, score 0.75, subj 0.3, obj 0.3)

## [AMBIGUOUS] #16
*Internal reflectors within MTDs can appear discontinuous in seismic data.*

- 1. `mass transport deposit` --[hasDescriptor]--> `continuous` (T1, score 1.75, subj 0.6, obj 1.0)
- 2. `mass transport deposit` --[hasDescriptor]--> `discontinuous` (T2, score 1.75, subj 0.6, obj 1.0)

## [AMBIGUOUS] #17
*Retrogressive failure may propagate upslope from the initial failure surface.*

- 1. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [AMBIGUOUS] #18
*Large submarine landslides commonly occur on passive continental margins.*

- 1. `submarine landslides` --[occursIn]--> `mediterranean continental margin` (T1, score 1.45, subj 1.0, obj 0.3)
- 2. `submarine landslides` --[occursIn]--> `mediterranean continental margins` (T1, score 1.45, subj 1.0, obj 0.3)
- 3. `submarine landslides` --[occursIn]--> `tectonically active margin` (T2, score 1.45, subj 1.0, obj 0.3)

## [PARTIAL_ONLY] #19
*Sediment failure can occur along passive continental margins.*

- 1. `mass transport deposit` --[occursIn]--> `continental margin` (T1, score 1.15, subj 0.0, obj 1.0)
- 2. `turbidite system` --[occursIn]--> `continental margin` (T1, score 1.15, subj 0.0, obj 1.0)
- 3. `rapid sediment accumulation` --[causes]--> `failure` (T1, score 1.0, subj 0.0, obj 1.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #20
*Internally generated seepage forces may fracture the seabed and contribute to slope instability.*

- 1. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [AMBIGUOUS] #21
*Steep slopes may promote seabed fracturing and slope failure.*

- 1. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [AMBIGUOUS] #22
*Debris flows may occur on continental slopes.*

- 1. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [PARTIAL_ONLY] #23
*Debris flows may display chaotic seismic facies.*

- 1. `debris-flow deposit` --[hasDescriptor]--> `chaotic` (T1, score 1.15, subj 0.0, obj 1.0)
- 2. `megaslide` --[hasDescriptor]--> `chaotic` (T1, score 1.15, subj 0.0, obj 1.0)
- 3. `unit a` --[hasDescriptor]--> `chaotic` (T1, score 1.15, subj 0.0, obj 1.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [PARTIAL_ONLY] #24
*Turbidity currents can be generated from debris flows.*

- 1. `rough topography` --[controls]--> `disintegration and transformation of the debris flows into a turbidity current` (T1, score 0.3, subj 0.0, obj 0.3)
- 2. `submarine landslides` --[affects]--> `subsequent flows` (T2, score 0.3, subj 0.0, obj 0.3)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #25
*Elevated pore pressure can reduce slope stability and promote slope failure.*

- 1. `excess pore pressure` --[causes]--> `slope failure` (T2, score 1.3, subj 0.3, obj 1.0)
- 2. `excess pore pressure` --[causes]--> `slope failures` (T2, score 0.9, subj 0.3, obj 0.6)
- 3. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [PARTIAL_ONLY] #26
*Preserved sediment blocks within MTDs may display sub-horizontal internal reflectors.*

- 1. `channel levee complex` --[hasDescriptor]--> `thin` (T1, score 1.15, subj 0.0, obj 1.0)
- 2. `mass transport deposit` --[hasDescriptor]--> `low-amplitude` (T1, score 0.75, subj 0.6, obj 0.0)
- 3. `mass transport deposit` --[hasDescriptor]--> `high-amplitude` (T1, score 0.75, subj 0.6, obj 0.0)
- *(no triple has BOTH ends present; the above are single-end matches — check whether the statement paraphrases one of them)*

## [AMBIGUOUS] #27
*Fluid overpressure may contribute to slope failure in submarine environments.*

- 1. `excess pore pressure` --[causes]--> `slope failure` (T2, score 1.3, subj 0.3, obj 1.0)
- 2. `excess pore pressure` --[causes]--> `slope failures` (T2, score 0.9, subj 0.3, obj 0.6)
- 3. `sea floor slope` --[affects]--> `effect of the surface slope` (T2, score 0.6, subj 0.3, obj 0.3)

## [AMBIGUOUS] #28
*Basal surfaces of MTDs can display disrupted reflectors.*

- 1. `mass transport deposit` --[partOf]--> `basal shear surface` (T1, score 0.9, subj 0.6, obj 0.3)
- 2. `mass transport deposit` --[partOf]--> `upper surface` (T1, score 0.9, subj 0.6, obj 0.3)

## [AMBIGUOUS] #29
*MTDs can exhibit discontinuous internal reflectors in seismic profiles.*

- 1. `mass transport deposit` --[hasDescriptor]--> `continuous` (T1, score 1.75, subj 0.6, obj 1.0)
- 2. `mass transport deposit` --[hasDescriptor]--> `discontinuous` (T2, score 1.75, subj 0.6, obj 1.0)

