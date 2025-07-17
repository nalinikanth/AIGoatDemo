# Adversarial Example Attack (Integrated Demo)

## What is an Adversarial Example?
Adversarial examples are inputs to machine learning models that have been intentionally perturbed in subtle ways to cause the model to make a mistake. These changes are often imperceptible to humans but can fool even state-of-the-art models.

---

## Where is it in AIGoat?
On the product image upload or test page, you can upload an image, generate an adversarial version, and see how the model’s prediction changes.

---

## Step-by-Step: Demo the Attack
1. **Go to the Adversarial Example Demo:**
   - Navigate to the "Security Lab" or the adversarial example test page.
2. **Upload an Image:**
   - Choose a product image or any image file.
   - The model will show its prediction for the original image.
3. **Generate Adversarial Image:**
   - Set the attack strength (epsilon) and click “Generate Adversarial Image.”
   - The adversarial image and its (likely incorrect) prediction will be shown.

---

## Mitigations
- **Adversarial Training:** Train models with adversarial examples to improve robustness.
- **Input Preprocessing:** Apply transformations to inputs to reduce the effect of adversarial noise.
- **Model Monitoring:** Detect and flag suspicious or out-of-distribution inputs.

---

## References
- [Explaining and Harnessing Adversarial Examples (Goodfellow et al.)](https://arxiv.org/abs/1412.6572)
- [Adversarial Examples in Machine Learning](https://en.wikipedia.org/wiki/Adversarial_machine_learning) 