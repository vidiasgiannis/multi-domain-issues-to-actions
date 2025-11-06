import os
try:
    import openai
except Exception:
    openai = None

def recommend_actions(issue_text, kb_snippets):
    context = "\n\n".join([f"[{n}]\n{t}" for n,t in kb_snippets])
    if openai and os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI()
        prompt = f"You are a Support/SRE assistant. Based on the issue and KB, output 3 concise, actionable steps.\nIssue:\n{issue_text}\n\nKB:\n{context}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    return "- Follow runbook checks\n- Scale/throttle affected component\n- Add guardrail; open RCA task"
