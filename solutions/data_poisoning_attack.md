# Data Poisoning Attack (Integrated Demo)

## What is Data Poisoning?
Data poisoning is an attack where an adversary injects malicious data into the training set of a machine learning model. The goal is to manipulate the model’s behavior, such as causing it to make incorrect predictions or recommendations.

---

## Where is it in AIGoat?
On any product page, the review form includes a “Simulate Malicious Review (Data Poisoning)” checkbox for users in Red Team Mode. This allows you to submit a review that poisons the recommendation model.

---

## Step-by-Step: Demo the Attack
1. **Enable Red Team Mode:**
   - Set `NEXT_PUBLIC_RED_TEAM_MODE=true` in your `.env.local` and restart the frontend.
2. **Go to a Product Page:**
   - Scroll to the review form.
   - In Red Team Mode, you’ll see a “Simulate Malicious Review” checkbox and a user field.
3. **Submit a Malicious Review:**
   - Check the box, set the user (e.g., `attacker`), and submit a high rating for the product.
   - You’ll see a confirmation and a button to “See Recommendation Impact for this User.”
4. **See the Impact:**
   - Click the button to immediately see how the poisoned review affects recommendations for the attacker.

---

## Mitigations
- **Data Validation:** Check and clean training data for anomalies or outliers.
- **Access Controls:** Restrict who can submit or modify training data.
- **Robust Training:** Use algorithms that are less sensitive to outliers or poisoned data.
- **Monitoring:** Continuously monitor model outputs for unexpected changes.

---

## References
- [Poisoning Attacks on Machine Learning](https://arxiv.org/abs/1804.00792)
- [OWASP ML Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)

