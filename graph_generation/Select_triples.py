import os
import re
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
def extract_data_in_brackets(input_string):
    # 一种正则表达式目的是提取[]里面的内容
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches


def generate_relationships(entity1: str, entity2: str):
    """
    Generate relationships between two specified entities. Return relationships in the form of triples. If no relationship is found, return an empty result.

    Args:
        entity1 (str): The head entity.
        entity2 (str): The tail entity.

    Returns:
        str: A string of relationships between the two entities or an empty result if no relationship is found.
    """
    # Define the prompt with specific entities
    generation_prompt = f"""
        Given the entities '{entity1}' and '{entity2}', generate  relationships  between them.
        The relationships should be helpful for healthcare prediction (e.g., drug recommendation, diag prediction, etc.).
        Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive and as short as possible.
        Generate the relationships in a detailed manner and provide a comprehensive list.
        If no relationship can be identified between these entities, return an empty result or indicate that no relationship exists.
        All generated relationships should be in English and formatted as triples.
    """

    # Call the chat API with the prompt
    chatgpt = ChatGPT()
    response = chatgpt.chat(generation_prompt)

    # Extract and format the response
    json_string = str(response)
    triples = extract_data_in_brackets(json_string)

    if not triples:
        return ""  # Return empty if no relationships found

    outstr = ""
    for triple in triples:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'

    return outstr


# Usage example
entity1 = "Acute myocardial infarction"
entity2 = "Cardiac dysrhythmias"
result = generate_relationships(entity1, entity2)
print(result)

def main():
    # Usage example
    entity1 = "Acute myocardial infarction"
    entity2 = "Cardiac dysrhythmias"
    result = generate_relationships(entity1, entity2)
    print(result)
if __name__ == '__main__':
    main()