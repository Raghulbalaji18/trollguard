import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Download VADER lexicon
nltk.download("vader_lexicon")

print("🧠 Loading NLP + AI models... please wait.")
# ML/NLP pipelines
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
sarcasm_model = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")

# NLP-based sentiment analyzer
vader = SentimentIntensityAnalyzer()

def analyze_text(text):
    print("\n--- Analyzing Text ---")
    print(f"📝 Input: {text}\n")

    # 1️⃣ Toxicity
    tox_results = toxicity_model(text)[0]
    toxic_score = [r for r in tox_results if r['label'] == 'toxic'][0]['score']

    # 2️⃣ Emotion
    emo_results = emotion_model(text)[0]
    emotion = sorted(emo_results, key=lambda x: x['score'], reverse=True)[0]

    # 3️⃣ Sarcasm
    sarcasm_result = sarcasm_model(text)[0]['label']
    sarcasm_detected = "sarcasm" in sarcasm_result.lower()

    # 4️⃣ NLP Sentiment (VADER)
    vader_score = vader.polarity_scores(text)
    compound = vader_score['compound']

    # 🧩 Combined Logic
    print("🔹 Toxicity:", round(toxic_score, 2))
    print("🔹 Emotion:", emotion['label'].capitalize(), "-", round(emotion['score'], 2))
    print("🔹 Sarcasm:", "Yes" if sarcasm_detected else "No")
    print("🔹 Sentiment (VADER):", compound)

    # Decision Fusion
    if toxic_score > 0.6 or compound < -0.5:
        classification = "Toxic / Abusive"
    elif sarcasm_detected:
        classification = "Sarcastic"
    elif emotion['label'] in ["disgust", "anger"]:
        classification = "Disgust / Angry"
    elif compound >= 0.3:
        classification = "Positive / Normal"
    else:
        classification = "Neutral / Normal"

    confidence = (
        max(toxic_score, emotion['score'], abs(compound))
        * (1.2 if sarcasm_detected else 1)
    )
    confidence = min(confidence * 100, 99.9)

    print(f"\n🧠 Final Classification: {classification}")
    print(f"🔒 Confidence: {confidence:.1f}%")
    print("-" * 60)


if __name__ == "__main__":
    print("\n🤖 Welcome to TrollGuard 3.0 – Hybrid NLP Tone Analyzer")
    print("Type text to analyze (type 'exit' to quit)\n")

    while True:
        user_text = input("Enter text: ").strip()
        if user_text.lower() == "exit":
            print("\n👋 Exiting TrollGuard 3.0. Stay kind online!")
            break
        elif not user_text:
            print("⚠️ Please enter some text.")
        else:
            analyze_text(user_text)
