# Rule Hiding by Evidence — Repository Guide

This repository provides the full implementation, data, and prompts for the study “Rule Hiding by Evidence: Regulated Predictors in Decision-Tree Models.” The pipeline harvests decision-tree papers, extracts explicitly reported predictors, maps them to a regulated data category ontology, links them to paragraph-cited privacy laws, and applies audit-based corrections reported in the study.

## Repository layout
- `data/raw/` — harvested Crossref/OpenAlex metadata per industry query.
- `data/processed/` — outputs from every pipeline stage (merged corpus, relevance and domain labels, predictor tables, regulatory clauses, validator results, high-confidence regulated pairs).
- `data/processed/audited/` — human-audit samples used to estimate stage-wise precision and the compound multiplier \(S\).
- `src/` — pipeline code:
  - `harvest/` — data collection.
  - `transform/` — LLM-driven processing stages.
  - `validation/` — audit calculations and multiplier computation (no API calls).
- `notebooks/Results_Analysis.ipynb` — regenerates the figures and tables used in the study.

## Pipeline (method-aligned)
1. **Harvest** (`config/queries.yaml`, `src/harvest/*.py`): Crossref/OpenAlex queries for 12 industry sectors (1,000 per source/sector) → `data/raw/crossref_<domain>.json`, `data/raw/openalex_<domain>.json`.
2. **Merge & deduplicate** (`src/transform/step_1.merge_dedup_enhanced.py`): DOI-priority merge with title+year backstop → `data/processed/merged_corpus.{json,csv}` (19,405 records).
3. **Decision-tree relevance** (`step_2.classify_decision_trees.py`): LLM binary screen → `merged_classified_decision_trees_related.json`.
4. **Industry assignment** (`step_3.classify_domain.py`): 13-sector classifier on relevant items → `merged_domain_validated.{json,csv}` (8,386 relevant; 4,686 retained where searched = assigned sector).
5. **Predictor extraction** (`step_4.extract_features.py`): explicit decision-tree predictors with evidence sentences → `merged_features.{json,csv}` for the 4,686 retained papers.
6. **Predictor validation** (`step_5.validate_features.py`): strict gate; 596 DOIs retained as `feature_validation=="Valid"` → `validated_features.{json,csv}`.
7. **Attribute-class mapping** (`step_6.auto_attribute_class.py`): map validated predictors to 13 regulated data categories → `attribute_classes.{json,csv}` (2,171 predictor rows; 1,749 unique predictors).
8. **Regulatory fragment catalog** (`step_7.data_extraction_from_regulations.py`): tag paragraph-level snippets from 13 frameworks (GDPR, ePrivacy, NIS2, PSD2, EU eHealth Network, CCPA, CPRA, HIPAA, HITECH, GLBA, COPPA, FERPA, ECPA) → `reg_sections_clauses.{json,csv}`.
9. **Predictor–regulation validation** (`step_8.validate_feature_regulation.py`): pair predictors with regulations sharing the attribute class; LLM validation of status and confidence → `validated_feature_regulation.{json,csv}` (9,256 pairs).
10. **High-confidence reporting set** (`step_9.filter_regulated.py`): retain only `Regulated` + `High` → `validated_feature_regulation_regulated.json` (2,329 pairs). Analyses apply the audit multiplier \(S = 0.415065\).
11. **Audits** (`src/validation/step_*.py`, `data/processed/audited/`): stratified human labels for relevance, domain, predictor validity, attribute class, and regulation status; scripts compute stage-wise precision and the compound multiplier \(S\).
12. **Analysis** (`notebooks/Results_Analysis.ipynb`): regenerates figures and tables from the processed data.

## Reproducibility and dependencies
- Python ≥ 3.11; install with `pip install -r requirements.txt`.
- Azure OpenAI credentials are required for the LLM-driven steps in `src/transform`:
  - `export OPENAI_API_BASE=...`
  - `export OPENAI_API_KEY=...`
- Audit scripts in `src/validation` run on the checked-in audited datasets and do not issue API calls.
- All processed artifacts are versioned in `data/processed/`; figures can be regenerated via the analysis notebook without rerunning the API-dependent steps.

## How to run
1. Harvest: `python src/harvest/1.crossref.py` and `python src/harvest/2.openalex.py`
2. Transform (LLM) steps in order 1–9: `python src/transform/step_<n>.*.py`
3. Optional audits: `python src/validation/step_*.py` to recompute precision multipliers and \(S\)
4. Analysis: open `notebooks/Results_Analysis.ipynb`

## Key artifacts
- Attribute-classed predictors: `data/processed/attribute_classes.json`
- Regulated, high-confidence pairs (pre-audit scaling): `data/processed/validated_feature_regulation_regulated.json`
- Audit inputs and multiplier computation: `data/processed/audited/*`, `src/validation/step_6_final_progated_error.py`
- Figures and tables: regenerated via `notebooks/Results_Analysis.ipynb`

## Notes
- Controlled vocabularies: 12 industry sectors and 13 regulated data categories.
- Reported counts are reproducible from the checked-in processed data (19,405 harvested records; 4,686 sector-aligned; 596 predictor-valid DOIs; 2,329 Regulated+High pairs before applying \(S\)).
