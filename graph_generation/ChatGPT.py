import json
import os
from openai import OpenAI

# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-qLsrNRqfAjJ0uqYq3fD46a9b2309427d9d9664196236EaBc"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        self.client = OpenAI()
        self.messages = []
    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages
        )
        # self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        context = response.choices[0].message.content
        return context
if __name__ == '__main__':
    chat = ChatGPT()
    example = """
        Example:
        prompt: systemic lupus erythematosus
        updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for, systemic lupus erythematosus], [systemic lupus erythematosus, is treated with, steroids], [methylprednisolone, is a, steroid]]
    """
    prompt_message = f"""
        Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.
        The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
        Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
        Both ENTITY 1 and ENTITY 2 should be noun.
        Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
        Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 100.

        {example}

        prompt: {"iobenzamic acid"}
        updates:
    """
    response = chat.chat(prompt_message)
    print("Assistant Response:", response)

    # # Handle potential NaN values
    # try:
    #     response_obj = json.loads(response)
    #     response_obj = chat.handle_nan(response_obj)
    #     json_string = json.dumps(response_obj)
    #     with open('response.json', 'w') as json_file:
    #         json_file.write(json_string)
    # except ValueError as e:
    #     print(f"Error while processing JSON: {e}")
