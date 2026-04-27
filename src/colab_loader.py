import pandas as pd
import json


class ColabResultsLoader:
    def __init__(self, base_path, ft_path):
        self.base_df = pd.read_csv(base_path)
        self.ft_df = pd.read_csv(ft_path)

        # Normalize clause text
        self.base_df["clause"] = self.base_df["clause"].astype(str).str.strip()
        self.ft_df["clause"] = self.ft_df["clause"].astype(str).str.strip()

        # Convert to lists (better for fuzzy search)
        self.base_clauses = self.base_df["clause"].tolist()
        self.ft_clauses = self.ft_df["clause"].tolist()

        print(f"Loaded {len(self.base_df)} base samples")
        print(f"Loaded {len(self.ft_df)} fine-tuned samples")

    # -----------------------------
    # 🔥 FUZZY MATCH FUNCTION (KEY FIX)
    # -----------------------------
    def find_match(self, clause, clauses_list):
        clause = clause.strip()

        for c in clauses_list:
            if clause == c:
                return c

        for c in clauses_list:
            if clause in c or c in clause:
                return c

        return None

    # -----------------------------
    # RAW OUTPUT ACCESS (FIXED)
    # -----------------------------
    def get_base_output(self, clause):
        match = self.find_match(clause, self.base_clauses)

        if match is None:
            print("❌ BASE MATCH NOT FOUND")
            return "NOT_FOUND"

        return self.base_df[self.base_df["clause"] == match]["output"].values[0]


    def get_ft_output(self, clause):
        match = self.find_match(clause, self.ft_clauses)

        if match is None:
            print("❌ FT MATCH NOT FOUND")
            return "NOT_FOUND"

        return self.ft_df[self.ft_df["clause"] == match]["output"].values[0]

    # -----------------------------
    # JSON EXTRACTION (IMPROVED)
    # -----------------------------
    def extract_json(self, text):
        if text == "NOT_FOUND":
            return None

        try:
            blocks = []

            for i in range(len(text)):
                if text[i] == '{':
                    count = 0
                    for j in range(i, len(text)):
                        if text[j] == '{':
                            count += 1
                        elif text[j] == '}':
                            count -= 1
                        if count == 0:
                            blocks.append(text[i:j+1])
                            break

            for block in reversed(blocks):
                try:
                    return json.loads(block)
                except Exception:
                    continue

            # 🔥 smarter fallback
            text_lower = text.lower()

            if "share" in text_lower or "third party" in text_lower:
                risk = "Predatory"
            elif "right" in text_lower or "delete" in text_lower or "access" in text_lower:
                risk = "Fair"
            else:
                risk = "Unknown"

            return {
                "risk_status": risk,
                "dark_pattern_category": "General Terms",
                "explanation": text[:200],
                "violated_statute": "None"
            }

        except Exception:
            return None

    # -----------------------------
    # CLEAN JSON ACCESS
    # -----------------------------
    def get_base_json(self, clause):
        raw = self.get_base_output(clause)
        return self.extract_json(raw)


    def get_ft_json(self, clause):
        raw = self.get_ft_output(clause)
        return self.extract_json(raw)