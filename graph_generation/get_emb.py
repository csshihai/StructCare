# import requests
# import json
import time

# with open("../../resources/openai.key", 'r') as f:
#     key = f.readlines()[0][:-1]
#
# def embedding_retriever(term):
#     # Set up the API endpoint URL and request headers
#     url = "https://api.openai.com/v1/embeddings"
#     # 请求头
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {key}"
#     }
#
#     # 使用要嵌入的文本字符串和要使用的模型设置请求有效负载（请求主体）
#     payload = {
#         "input": term,
#         "model": "text-embedding-ada-002"
#     }
#
#     # 发送请求并检索响应
#     response = requests.post(url, headers=headers, data=json.dumps(payload))
#
#     # 从响应 JSON 中提取文本嵌入
#     embedding = response.json()["data"][0]['embedding']
#
#     return embedding


import requests
import json


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=08LqOHzVJPXQSBziFLqJOG1v&client_secret=RYUClCkYWRWrwR3vRcWRKfLZwQn7oDOp"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def embedding_retriever(term):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" +get_access_token()

    payload = json.dumps({
        "input":[term]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    embedding = None
    while embedding is None:
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            response_data = response.json()
            if 'error_code' in response_data and response_data['error_code'] == 18:
                print("API rate limit reached. Waiting before retrying...")
                time.sleep(5)  # 等待5秒钟后重试
                continue  # 跳过本次循环，重试
            embedding = response_data["data"][0]['embedding']
        except KeyError as e:
            print(response.json())
            print(f"Error occurred: {e}. JSON response does not contain the expected key.")
            # 记录错误或返回默认值
            embedding = []  # 或者其他适当的默认值

    return embedding


def main():
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" +get_access_token()

    payload = json.dumps({
        "input":["coordination between healthcare teams"]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    embedding = response.json()["data"][0]['embedding']
    print(embedding)

if __name__ == '__main__':
    main()