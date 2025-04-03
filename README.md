# Human Emotion Recognition using Gaussian Mixture Model (GMM)

## 📌 Overview

This project aims to classify human emotions—**Happy, Sad, Angry, Neutral**—from voice recordings using the **Gaussian Mixture Model (GMM)**. The primary application areas include:

- **AI-Powered Virtual Assistants** 🎙️
- **Mental Health Monitoring** 🏥
- **Human-Computer Interaction & Voice AI** 🤖

## 🚀 Why Gaussian Mixture Model (GMM)?

**Captures Variability**: Voice signals are continuous and can be effectively modeled as a mixture of Gaussian distributions.
**Handles Multi-Modal Distributions**: Unlike traditional classifiers, GMM can model variations within the same emotion class.
**Personalized AI Assistants**: Helps in building adaptive AI that can understand user emotions in real-time. 
**Healthcare Applications**: Supports mental health tracking by analyzing emotional patterns in speech.

## 📂 Dataset: RAVDESS Emotional Speech Audio Dataset

We utilize the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains **professional-quality emotional speech recordings**. This dataset provides diverse emotions expressed through speech, making it ideal for emotion recognition tasks.

## 🔧 How It Works?

1️⃣ **Feature Extraction**: Extract **Mel-Frequency Cepstral Coefficients (MFCCs)** from speech signals. 
2️⃣ **Modeling Emotions**: Use **Gaussian Mixture Model (GMM)** to cluster voice patterns into emotion categories. 
3️⃣ **Gaussian Components**: Fit Gaussian distributions to capture variations in **tone, pitch, and frequency**. 
4️⃣ **Classification**: Predict emotions using **maximum likelihood estimation**.

## 🛠️ Tech Stack

- **Python** 🐍
- **Librosa** (Audio Processing) 🎵
- **Scikit-learn** (Machine Learning) 🤖
- **NumPy & Pandas** (Data Handling) 📊
- **Matplotlib & Seaborn** (Data Visualization) 📈

## 📊 Expected Output

The model classifies voice recordings into one of the four emotions (**Happy, Sad, Angry, Neutral**) with high accuracy. The final results can be visualized through **confusion matrices and emotion distribution plots**.

## 🔮 Future Enhancements

- Expand the model to **more emotions** (Fear, Surprise, Disgust).
- Integrate with **real-time emotion detection systems**.
- Fine-tune using **deep learning techniques (LSTMs, CNNs on spectrograms)**.

## 📝 How to Use

1. Install dependencies:
   ```bash
   pip install numpy pandas librosa scikit-learn matplotlib seaborn
   ```
2. Run the script to extract features and train the GMM model.
3. Use the trained model to classify new voice recordings.

---

🚀 **Contributions & Feedback**: Feel free to contribute and suggest improvements! Let's make AI more emotionally intelligent! 😊

