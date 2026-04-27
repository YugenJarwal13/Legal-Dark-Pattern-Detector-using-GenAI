def base_prompt(text):
    return f"""
You are a legal AI system.

Analyze the clause:
\"\"\"{text}\"\"\"

Return ONLY valid JSON:
{{
  "risk_status": "...",
  "dark_pattern_category": "...",
  "explanation": "...",
  "violated_statute": "..."
}}

Do not add any text outside JSON.
"""


def few_shot_prompt(text):
    return f"""
You are a legal AI system.

Example 1:
Clause:
"We may share your data with partners."
Output:
{{
  "risk_status": "High",
  "dark_pattern_category": "Data Sharing",
  "explanation": "The clause allows broad sharing of user data without clear consent.",
  "violated_statute": "GDPR Article 6"
}}

Example 2:
Clause:
"You may request deletion of your data at any time."
Output:
{{
  "risk_status": "Fair",
  "dark_pattern_category": "User Rights",
  "explanation": "This clause supports user control over personal data.",
  "violated_statute": "None"
}}

Now analyze:

Clause:
\"\"\"{text}\"\"\"

Return ONLY valid JSON:
{{
  "risk_status": "...",
  "dark_pattern_category": "...",
  "explanation": "...",
  "violated_statute": "..."
}}

Do not add any text outside JSON.
"""


def rag_prompt(text, context_list):
    context = "\n\n".join(context_list)

    return f"""
You are an expert legal AI system specializing in GDPR compliance analysis.

Your task is to analyze a Terms of Service clause and determine whether it is fair or predatory, using GDPR legal principles.

You MUST base your reasoning on the provided GDPR context.

----------------------------------------
GDPR CONTEXT (Use this for reasoning):
{context}
----------------------------------------

INSTRUCTIONS:

1. Carefully read the clause.
2. Identify if it violates user rights, transparency, consent, or data processing rules.
3. Use the GDPR CONTEXT to justify your reasoning natively (e.g., "Under GDPR, data subjects must be informed..."). Do NOT artificially say "According to GDPR CONTEXT".
4. If relevant, explicitly reference GDPR Articles strictly in the format "GDPR Article X" (e.g., "GDPR Article 6", "GDPR Article 13").
5. Do NOT rely on general knowledge — prioritize the given context.
6. Be precise and legally grounded.
7. If multiple Articles apply, mention all relevant ones using the "GDPR Article X" format.
8. Do not repeat the clause verbatim in your explanation.
9. Avoid over-claiming (e.g., do not definitively state a clause 'lacks consent' if it merely fails to specify a lawful basis).
10. Prioritize specific operational Articles (e.g., GDPR Article 6 for lawful basis, GDPR Article 13 for information provision) over generic principles like GDPR Article 5.
11. Always prefix statutes with "GDPR Article". Example: "GDPR Article 13"
12. If the provided GDPR context is not sufficient, use best legal reasoning but clearly state the limitation internally (do not mention in output).

----------------------------------------
CLASSIFICATION RULES:

- Predatory → violates GDPR principles (lack of consent, unclear data use, forced agreement, hidden terms)
- Fair → supports user rights (access, deletion, transparency, consent)

----------------------------------------
CLAUSE:
\"\"\"{text}\"\"\"

----------------------------------------
OUTPUT FORMAT (STRICT):

Return ONLY valid JSON:

{{
  "risk_status": "Predatory or Fair",
  "dark_pattern_category": "Return ONLY ONE category from: Data Sharing, User Rights, Tracking & Privacy, Transparency, General Terms",
  "explanation": "Clear legal reasoning using GDPR context, written naturally.",
  "violated_statute": "Strict format: 'GDPR Article X' (e.g., 'GDPR Article 13'), else 'None'"
}}

DO NOT add any text outside JSON.
"""