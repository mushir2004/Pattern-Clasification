Human Emotion Recognition using Gaussian Mixture Model (GMM)

Problem Statement:
Classify human emotions (Happy, Sad, Angry, Neutral) from voice recordings using GMM.

Use Case: AI-Powered Virtual Assistants & Mental Health Monitoring
•	Industry: Healthcare, Human-Computer Interaction, Voice AI

•	Why GMM?
o	Voice signals are continuous and can be modeled as a mixture of Gaussian distributions.
o	Unlike traditional classifiers, GMM can model multi-modal distributions (e.g., variations in the same emotion).
o	Helps in personalized AI assistants and mental health applications.

Dataset: RAVDESS Emotional Speech Audio Dataset

How it Works?
1.	Extract MFCC Features from speech signals.
2.	Use GMM to cluster similar voice patterns into emotion categories.
3.	Fit Gaussian Components to capture variations in tone, pitch, and frequency.
4.	Classify Speech Emotions based on maximum likelihood.

