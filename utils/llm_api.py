from openai import OpenAI

from utils import api_base, api_keys


def ask(user_prompt, system_prompt=None, temperature=0.0, model="gpt-3.5-turbo") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    newclient = OpenAI(api_key=api_keys, base_url=api_base)

    completion = newclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    answer = completion.choices[0].message.content

    return answer
