# Folklore ML Lab agent guide

## Repository status

Preserved experiment machinery; no new experiment is active. Existing checked-in
runs remain frozen to Corpus v0.1.0, while the live v0.2.1 lock governs tooling
and newly generated outputs. Future ML work requires a separately approved,
evidence-backed need.

## Delegated execution

An owner-approved roadmap or phase authorizes every action explicitly listed in
that scope end-to-end. Do not request reconfirmation for settled decisions.
Finish all unblocked work before returning at the defined checkpoint.

Interrupt a phase only when a hard external blocker prevents progress, evidence
invalidates the phase premise, an unresolved choice would materially change the
owner-visible outcome, or completion requires a destructive or out-of-scope
action that was not authorized. Minor uncertainty is not a blocker: decide,
document, and continue. Record new ideas for the checkpoint; do not silently
expand scope or begin the next unapproved phase.

## Product evidence

Keep these states distinct: owner-confirmed need, observed behavior, agent
hypothesis, technically implemented, and human-validated. Plans and earlier
agent recommendations are context unless the owner approves them. CI, stubs,
simulated reviewers, and generated examples prove mechanics, not owner
usefulness. Agents may propose domain topics or product choices but may not
select them on the owner's behalf unless that choice was explicitly delegated.
