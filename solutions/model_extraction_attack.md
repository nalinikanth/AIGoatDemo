# Model Extraction (Stealing) Attack (Integrated Demo)

## What is a Model Extraction Attack?
A model extraction (or stealing) attack occurs when an adversary reconstructs or approximates a machine learning model by repeatedly querying it and collecting input/output pairs. This can lead to intellectual property theft or further attacks.

---

## Where is it in AIGoat?
On the recommendations page (or in the Security Lab), admins/red teamers can view and download the log of all recommendation queries and responses, simulating how an attacker would steal the model.

---

## Step-by-Step: Demo the Attack
1. **Go to the Recommendations Page or Security Lab:**
   - Query the recommender model with different users.
2. **Download the Extraction Log:**
   - Use the “Download Log” button to get all input/output pairs.
3. **(Optional) Train a Surrogate Model:**
   - Use the downloaded log to train a surrogate model outside the app (for advanced users).
4. **Red Team Mode:**
   - If enabled, a challenge banner may appear, explaining the risk and encouraging you to try to reconstruct the model.

---

## Mitigations
- **Rate Limiting:** Limit the number of queries per user/IP.
- **Output Perturbation:** Add noise to model outputs.
- **Watermarking:** Embed unique patterns in model responses to detect stolen models.
- **Monitoring:** Detect and block suspicious query patterns.

---

## References
- [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)
- [OWASP ML Top 10](https://owasp.org/www-project-machine-learning-security-top-10/) 