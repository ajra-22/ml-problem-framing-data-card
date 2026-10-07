# ML Problem Framing & Risk Register

## 1. Problem Framing Memo
* **Business Decision:** Decide whether to trigger proactive customer retention interventions (e.g., targeted loyalty offers, dedicated account management outreach).
* **Prediction Target:** Binary classification `churned` (1 = Customer canceled subscription, 0 = Customer remains active).
* **Unit of Observation:** Individual customer account (`customer_id`).
* **Action Window:** 30-day decision cycle prior to monthly renewal billing.
* **Non-ML Baseline:** Rule-based heuristic flagging accounts where `last_login_days > 10` AND `support_tickets >= 3`.

## 2. Business & Model Metrics
* **False Positive Cost:** Low to Moderate (cost of unnecessary discount/outreach to a loyal customer).
* **False Negative Cost:** High (loss of recurring revenue and increased customer acquisition cost).
* **Primary Metrics:** Recall, Precision, F1-Score, and ROC-AUC curve.

## 3. Governance, Abstention & Safeguards
* **Abstention Condition:** If model confidence score is between 0.40 and 0.60, refrain from automated intervention.
* **Human Review:** Flag high-tier (`Pro` plan) accounts with high churn risk for manual CSM (Customer Success Manager) review.
* **Monitoring & Rollback:** Monitor weekly drift in prediction distribution; trigger model rollback if precision drops below 70%.