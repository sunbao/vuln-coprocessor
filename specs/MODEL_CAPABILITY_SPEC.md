# Model Capability Spec

## Scope

This file defines what the trained model in this repository is for.

It fixes:

- the intended business objective
- what the model is expected to do
- what the model must not be treated as
- the preferred product landing points

This file is a stable project rule.
It is not a run log.

## Primary Objective

The trained model is a vulnerability explanation coprocessor.

Its purpose is to transform structured vulnerability facts into stable, grounded, human-readable analysis in Chinese.

The target output should explain:

- why a component is reported as affected
- how the current risk should be understood
- what remediation path is suggested
- whether upgrade is recommended

## Input Assumption

The model does not create the underlying vulnerability finding.

The finding must already exist in structured upstream data such as:

- component identity
- component version
- vulnerability identifier
- vulnerability name
- risk level
- CVSS score when available
- recommended version
- latest version
- source evidence and candidate match metadata

The model is an interpretation layer on top of these facts.

## Expected Capabilities

The trained model is expected to support these capability classes:

### 1. Vulnerability explanation

Given structured input facts, the model should explain:

- which component is affected
- which vulnerability is involved
- why the record should be treated as a match according to the provided evidence

### 2. Risk explanation

The model should turn structured severity data into readable risk language without inventing unsupported impact claims.

### 3. Remediation guidance

When the input contains version guidance, the model should provide practical remediation suggestions such as:

- upgrade recommendation
- validation or regression reminder
- fallback handling when a clear upgrade target is missing

### 4. Upgrade recommendation

The model should give a direct conclusion on whether upgrade is recommended, based only on the supplied facts.

### 5. Batch reporting support

The model output should be suitable for:

- vulnerability detail pages
- project-level vulnerability summaries
- analyst review drafts
- report generation workflows

## Non-Goals

The trained model must not be treated as any of the following:

### 1. Vulnerability discovery engine

The model does not scan code, dependencies, or binaries to discover vulnerabilities.

### 2. Matching source of truth

The model does not replace the business matching logic implemented through:

- source data tables
- version normalization logic
- code-side vulnerability/component applicability rules

### 3. Vulnerability database authority

The model is not the authoritative source for:

- vulnerability existence
- component-to-vulnerability applicability
- official remediation version truth

Those must remain anchored to upstream data and business rules.

### 4. Unbounded general assistant

The model is not being trained to answer arbitrary security questions outside the provided vulnerability-fact context.

## Product Boundary

The correct product positioning is:

- post-scan explanation layer
- analyst copilot
- reporting assistant

The incorrect product positioning is:

- scanner replacement
- rule-engine replacement
- source-of-truth database replacement

## Success Criteria

The project should judge the trained model successful only when it can do all of the following on a frozen non-overlapping validation set:

- preserve the required output structure
- remain grounded in provided facts
- avoid fabricated vulnerability/component relations
- provide actionable remediation wording when the input supports it
- stay stable across common npm and maven style samples

## Failure Conditions

The project should treat the model as not yet ready if any of these remain common:

- fabricated facts not present in input
- unsupported causal claims
- wrong upgrade conclusion despite clear version evidence
- output drift away from the required response structure
- inability to explain standard vulnerability samples from the supported data pipeline

## Preferred Landing Points

Once validated, the model is best applied in:

- vulnerability detail explanation panels
- project vulnerability triage views
- batch report generation
- analyst review prefill workflows

## Dependency On Data Quality

Model quality depends directly on:

- upstream data correctness
- export consistency
- non-overlapping validation design
- stable labeling rules across dataset versions

If these inputs are weak, training loss alone is not a valid signal of success.

## Source Alignment Rule

The model capability defined here assumes a post-match data pipeline.

That means:

- upstream systems determine the finding
- the exported dataset carries the structured evidence
- the model explains the finding instead of inventing the finding

Do not redefine the model as a detector without a separate project scope and a separate validation framework.
