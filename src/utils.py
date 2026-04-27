ALLOWED = {
    "Data Sharing",
    "User Rights",
    "Tracking & Privacy",
    "Transparency",
    "General Terms"
}

def safe_parse(text, loader):
    obj = loader.extract_json(text)
    return obj if obj is not None else {
        "risk_status": "Unknown",
        "dark_pattern_category": "General Terms",
        "explanation": "Model output not parseable.",
        "violated_statute": "None"
    }

def normalize_risk(x):
    if not x: return "Unknown"
    x = x.lower()
    if x in ["predatory", "high", "caution"]: return "Predatory"
    if x in ["fair", "low", "none"]: return "Fair"
    return "Unknown"

def normalize_category(cat, explanation=""):
    text = (cat or "") + " " + (explanation or "")
    text = text.lower()

    if "share" in text or "third party" in text:
        return "Data Sharing"

    if "consent" in text:
        return "User Rights"

    if "track" in text or "cookie" in text:
        return "Tracking & Privacy"

    if "inform" in text or "transparent" in text:
        return "Transparency"

    return "General Terms"

import re

PRIORITY = ["13", "6", "7", "5"]

def normalize_statute(statute_text):
    if not statute_text:
        return "None"

    articles = re.findall(r'\d+', str(statute_text))

    if not articles:
        return "None"

    # 🔥 dynamic scoring
    def score(a):
        if a == "13":
            return 3
        if a == "6":
            return 2
        if a == "7":
            return 2
        if a == "5":
            return 1
        return 0

    best = sorted(articles, key=score, reverse=True)[0]

    return f"GDPR Article {best}"

def confidence_score(result):
    score = 0

    if result["risk_status"] != "Unknown":
        score += 1
    if result["violated_statute"] != "None":
        score += 1
    if len(result["explanation"]) > 50:
        score += 1

    return score

def normalize_true_label(label):
    if not isinstance(label, str):
        return "Unknown"
    
    label = label.lower()

    if label in ["predatory", "high", "caution"]:
        return "Predatory"
    if label in ["fair", "low"]:
        return "Fair"

    return "Unknown"
