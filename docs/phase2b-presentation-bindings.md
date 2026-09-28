# Phase 2B: frozen meal presentation authority

Schema v13 adds `meal_plan_item_presentation_bindings`, keyed by the existing
`(plan_id, plan_item_id)` composite foreign key. Each optional row contains a
versioned JSON envelope and the plan creation timestamp. The envelope contains
only JSON primitives; authoritative decimals are strings. A SHA-256 digest
catches accidental corruption, not malicious writes by someone controlling the
SQLite file. Application validation remains the authority boundary. Update and
delete triggers protect historical rows from routine SQL mutation.

Migration creates an empty companion table transactionally. No existing plan,
item, draft, fact, history, event, dispatch, replacement, request, application,
or intake row is changed. Legacy items remain unbound. In particular, historical
`natural_quantity_text` is never parsed to generate reverse authority. Legacy
sweet-potato quarter-piece descriptions also remain unbound: users can still
report official servings, supported physical/count units, or fractions of the
exact plan, but an old visual description alone cannot authorize conversion.

`PresentationBinding` is frozen and validates:

- format version (currently 1) and presentation classification;
- selected display text against the plan item;
- exact official-serving multiplier against the plan item;
- source identity, full official serving definition, occurrence/snapshot IDs,
  content signature, and original station/concept context;
- calibration applicability when the classification is reversible.

`EXACT_AUTHORITATIVE` records Python's canonical rendering. Reverse official-unit
arithmetic still uses the authoritative nutrition record, not display text.
`DESCRIPTIVE_ONLY` stores selected wording and identity guards, but cannot contain
a calibration. `REVERSIBLE_CALIBRATED` stores the entire original calibration:
ID, version, provenance, unit names, aliases, and exact units-per-official-serving
ratios (including the optional whole-item relationship). An independent visual
quantity need not be stored: the exact frozen plan multiplier and frozen ratio
already determine it. No rounded display quantity is a conversion source.

The single existing registry entry is now
`phelps-baked-sweet-potato-quarter`, version 1. Its evidence is the operator's
Phelps observation on 2026-09-03; its original component and 1 Each guards are
unchanged. It retains 4 quarter pieces / 1 whole sweet potato per official
serving. No station restriction or utensil measurement was invented from that
observation. Every binding additionally retains the original occurrence's
station/concept as a reuse guard. Future station/utensil-specific calibrations
will need explicit applicability rules in the registry and corresponding
render-request context; this step does not claim such evidence exists.

The shared recommendation-to-plan adapter freezes the validated selection.
Manual, scheduled, immediate-request, and replacement save transactions insert
the binding immediately after its item. Low-level hand-built plans may remain
unbound; persistence never invents calibration from their text. Plan equality
includes the frozen binding. Dispatch/replacement retries already load persisted
plans/replies; they do not rerender. A conflicting second render is not an
update mechanism. Binding creation time lives in the companion row and does not
participate in plan equality; classification, evidence, aliases and relationships
do, because those are material authority.

Planned visual conversion uses only the exact frozen plan binding. Unbound,
descriptive-only, legacy, and unplanned visual quantities remain unresolved;
the current global registry cannot authorize a report retroactively.
Relative-to-plan arithmetic, Phase 2A attestation guards, optimizer arithmetic,
and nutrition calculations are unchanged.

Before a literal quantity can use that binding, reconciliation checks its whole
local food-quantity relationship. Number words may normalize to digits; an
indefinite article normalizes only with a countable unit or an unambiguous
one-each food noun. Surrounding
modifiers and other food words cannot be discarded to manufacture an exact
amount or move it to another food. A comma-separated amount can refer back to
one immediately preceding, unambiguous named food. Unknown phrasing remains a
clarification, and a rejected correction leaves an earlier validated draft fact
intact with its clarification pending through restart. Unplanned food identity
needs a distinctive literal local reference;
one shared token or an ambiguous menu alias does not authorize intake. Draft
quantity reuse requires local wording that points to one durable source and a
grounded target, rather than any mention of the word "same".

This is development-only. Production remains on v12 and its stable checkout.
A later controlled rollout must rehearse/approve v13 separately. Phase 2C should
build on the persisted relationship, retain format-1 decoding, and never derive
physical conversions from display strings. Registry versions must change when
calibration evidence/relationships change; historical binding values remain the
source for old plans regardless of the current registry.
