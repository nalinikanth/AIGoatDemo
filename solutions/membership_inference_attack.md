# Membership Inference Attack (Integrated Demo)

## What is a Membership Inference Attack?
A membership inference attack allows an adversary to determine whether a specific data point was part of a machine learning model’s training set. This can lead to privacy breaches, especially if the training data contains sensitive information.

---

## Where is it in AIGoat?
On the user profile page, there is a “Privacy Check: Is Your Data in the Model?” section. Users can check if their data (e.g., a product they interacted with) is likely in the model’s training set.

---

## Step-by-Step: Demo the Attack
1. **Go to the User Profile Page:**
   - Navigate to your profile in the app.
2. **Find the Privacy Check Section:**
   - Enter/select a product you have interacted with.
3. **Check Membership:**
   - Click “Check Membership.”
   - The model’s confidence and a privacy warning (if applicable) will be displayed.
4. **Red Team Mode:**
   - If enabled, a challenge banner will appear, encouraging you to infer if a user’s data was used to train the model.

---

## Mitigations
- **Regularization:** Use techniques like dropout or L2 regularization to reduce overfitting.
- **Differential Privacy:** Train models with differential privacy to limit information leakage.
- **Limit Model Output:** Return less granular confidence scores or only top predictions.

---

## References
- [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820)
- [OWASP ML Top 10](https://owasp.org/www-project-machine-learning-security-top-10/) 