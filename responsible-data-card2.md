# Responsible Data Card: Customer Churn Dataset

## Dataset purpose
* **Supported Decision:** Identification of active customer accounts at high risk of immediate churn to trigger proactive customer retention interventions.
* **Prohibited Decisions:** Automated account terminations, pricing discrimination, or denial of standard service features.

## Provenance and permission
* **Creation & Source:** Internal subscription management and CRM logs.
* **Authorized Users:** Internal analytics and data science team members.
* **Consent & Licensing:** Subject to internal data policy; customer IDs are pseudonymized (`C001`, `C002`, etc.).

## Population and representation
* **Represented Groups:** Active subscription accounts across `Basic`, `Standard`, and `Pro` tiers.
* **Missing Groups:** Trial users, non-registered visitors, and custom enterprise accounts.
* **Sampling Limitations & Imbalance:** Benchmark dataset contains 12 sample rows (5 churned, 7 active).

## Features and target
* **Features:** `customer_id`, `tenure_months`, `support_tickets`, `monthly_spend_inr`, `last_login_days`, `plan_type`.
* **Target Label:** `churned` (0 or 1).
* **Data Leakage & Sensitive Proxies:** No direct demographic features collected; feature windows strictly cut off prior to prediction timestamp.

## Quality checks
* **Missingness:** 0% missing values across all attributes.
* **Duplicates:** 0 duplicate rows or IDs.
* **Class Balance:** 7 active (`churned = 0`), 5 churned (`churned = 1`).
* **Train/Test Separation:** Stratified split required to maintain class proportions.

## Risks and safeguards
* **False Positives (FP):** Risk of redundant discounts to loyal accounts.
* **False Negatives (FN):** Risk of losing customers without engagement.
* **Safeguards:** Human-in-the-loop review for high-tier accounts and confidence thresholds before automated action.

## Intended evaluation
* **Baseline Non-ML Model:** Rule-based decision model.
* **Performance Measures:** Precision, Recall, F1-Score, ROC-AUC.
* **Error Analysis:** Error segmentation across different subscription tiers (`plan_type`).