
import os
import base64
import tempfile
from pathlib import Path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
math_messages = []

def validate_image_format(image_path):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension not in valid_extensions:
        raise ValueError(f"Unsupported file format! Please upload one of these formats: {', '.join(valid_extensions)}")
    return file_extension

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path, file_extension):
    global math_messages
    math_messages = []

    if not os.path.exists(image_path):
        raise FileNotFoundError("Image file not found! Please provide a valid file path.")

    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    os.makedirs(uploaded_file_dir, exist_ok=True)

    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    temp_image_path = os.path.join(uploaded_file_dir, f"tmp{os.urandom(16).hex()}.jpg")
    image.save(temp_image_path)

    base64_image = encode_image(temp_image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that extracts math problems from images, including handwritten or typed equations, and formats them as questions in plain text."
        },{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Extract any math-related content (equations, expressions, or problems) from this image and format it as a question in plain text (no LaTeX). If the content is an equation, assume the question is to solve it. If no math content is found, say 'No math content found in the image.'"
                }
            ]
        }],
        max_tokens=1000
    )

    os.remove(temp_image_path)
    return response.choices[0].message.content

def is_vague_math_request(text):
    vague_phrases = [
        "solve it", "do the math",
        "help me", "solve the math", "can you help me", "what's the answer",
        "calculate this", "what's the solution", "explain this"
    ]
    return any(phrase in text.lower() for phrase in vague_phrases)

def get_response(user_input, image_description=None):
    global math_messages
    if not math_messages:
        math_messages.append({
            "role": "system",
            "content": (
                "You are a helpful math assistant. Your expertise is strictly limited to math-related questions. "
                "If the user asks a math-related question, solve it step-by-step using the Thetawise format (plain text, no LaTeX). "
                "If the user provides a vague input like 'solve the math' or 'what is the math?', assume they want to solve the math problem extracted from the image. "
                "If no math problem is provided, respond with: 'Please provide a specific math-related question or upload an image with a math problem.'\n"
                "Thetawise format for math problems:\n"
                "1. Start with a brief explanation of the problem.\n"
                "2. Clearly explain each step of the solution using plain text.\n"
                "3. Use clear spacing and line breaks.\n"
                "4. End with the final answer in plain text (e.g., 'So, the solution is x = ...')."
            )
        })

    if is_vague_math_request(user_input) and image_description:
        user_input = f"Solve the following math problem: {image_description}"

    if user_input and image_description:
        user_input = f"{image_description}\n\n{user_input}"

    math_messages.append({"role": "user", "content": user_input})

    if len(math_messages) > 10:
        math_messages = math_messages[-10:]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=math_messages,
        max_tokens=1000
    )

    answer = response.choices[0].message.content

    lines = answer.splitlines()
    deduped_lines = []
    prev_line = ""
    for line in lines:
        if line.strip() != prev_line.strip():
            deduped_lines.append(line)
            prev_line = line

    final_answer = "\n".join(deduped_lines)
    math_messages.append({"role": "assistant", "content": final_answer})
    return final_answer

def main():
    print("Welcome to Math Problem Solver!")
    print("You can: (1) Upload an image with a math problem, (2) Type a math question.")
    print("Type 'exit' or 'quit' to close the program.\n")

    global math_messages
    math_messages = []

    while True:
        print("How would you like to provide your math question?")
        print("(1) Upload an image")
        print("(2) Type a question")
        choice = input("Enter your choice (1/2, or 'exit' to quit): ").strip()

        if choice.lower() in ['exit', 'quit']:
            print("Exiting the program. Goodbye!")
            break

        image_description = None
        user_input = None

        if choice == '1':
            image_path = input("Provide the path to the image file (Supported formats: JPEG, PNG; press Enter if none): ").strip()
            image_path = image_path.strip('"').strip("'").replace('\\', '/')
            if not image_path:
                print("No image path provided. Please try again.")
                continue

            try:
                file_extension = validate_image_format(image_path)
                image_description = process_image(image_path, file_extension)
                print("\nMath content extracted from image:\n")
                print(image_description)
            except Exception as e:
                print("Error processing image:", e)
                continue

        elif choice == '2':
            user_input = input("\nWrite your math question: ").strip()
            if not user_input:
                print("Error: Please provide a valid math question.")
                continue

        else:
            print("Invalid choice! Please select 1 or 2.")
            continue

        if not user_input and not image_description:
            print("Error: Please provide a valid math question or an image with a math problem.")
            continue

        try:
            if user_input:
                answer = get_response(user_input, image_description)
                print("\nAI:\n")
                print(answer)
        except Exception as e:
            print("Error getting answer:", e)

if __name__ == "__main__":
    main()
