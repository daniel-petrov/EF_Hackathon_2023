from openai import OpenAI
from dotenv import load_dotenv

def generate_cover_letter(cv, job_desc):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Create a cover letter based on the following CV and job description."},
            {"role": "user", "content": f"CV:\n{cv}\n\nJob Description:\n{job_desc}"}
        ]
    )

    cover_letter = response.choices[0].message.content
    return cover_letter

def save_cover_letter(cover_letter, name = 'cover_letter.txt'):
    with open(name, "w") as f:
        f.write(cover_letter)