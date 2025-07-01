import openai

def explain_issue(field_name, value, issue):
    prompt = f"""You are a data validation assistant. A user has input the following:
Field: {field_name}
Value: {value}
Issue: {issue}
Explain in simple terms what is wrong and suggest how to fix it."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']