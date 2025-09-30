import json
import time
import requests
from openai import OpenAI
import os

# 关闭代理（避免影响本地请求）
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

# 初始化客户端
client = OpenAI(api_key="None", base_url="http://127.0.0.1:3000/v1")

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "state": {"type": "string", "description": "State abbreviation"},
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

# 本地工具函数（真实调用 wttr.in API）
def get_current_weather(city: str, state: str, unit: str):
    unit_param = "m" if unit == "celsius" else "u"
    url = f"https://wttr.in/{city}?format=3&{unit_param}"
    start = time.time()
    proxies = {
        "http": "http://10.186.163.27:8888",
        "https": "http://10.186.163.27:8888",
    }
    resp = requests.get(url, timeout=10, proxies=proxies)
    end = time.time()
    if resp.status_code == 200:
        return f"{resp.text.strip()}  (API call took {end - start:.3f}s)"
    else:
        return f"Failed to fetch weather (status {resp.status_code})"

tool_registry = {
    "get_current_weather": get_current_weather
}

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you."
    }
]

model_name = client.models.list().data[0].id

# ====== 统计量 ======
total_time = 0.0
llm_time = 0.0
func_time = 0.0
total_tokens = 0
func_tokens = 0

# ====== 主循环 ======
while True:
    # ---- 大模型推理时间 ----
    t0 = time.time()
    response_stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=0.95,
        max_tokens=1024,
        stream=True,  # Enable streaming
        tools=tools,
    )

    texts = ""
    tool_calls = []
    argument_fragments = []
    tool_name = None

    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            texts += chunk.choices[0].delta.content
        if chunk.choices[0].delta.tool_calls:
            tc = chunk.choices[0].delta.tool_calls[0]
            tool_calls.append(tc)
            if tc.function.name:
                tool_name = tc.function.name
            if tc.function.arguments:
                argument_fragments.append(tc.function.arguments)

    print("==== Text ====")
    print(texts)

    print("==== Tool Call ====")
    for tool_call in tool_calls:
        print(tool_call)

    # 拼接完整 JSON 参数
    full_arguments = "".join(argument_fragments).strip()
    print(f"streamed function call name: {tool_name}")
    print(f"streamed function call arguments: {full_arguments}")

    if not full_arguments:
        print("[Warning] No arguments collected from stream, skipping...")
        break

    try:
        tool_args = json.loads(full_arguments)
    except json.JSONDecodeError as e:
        print(f"[Error] JSONDecodeError: {e}")
        print(f"Raw arguments: {full_arguments}")
        break

    t1 = time.time()
    step_llm_time = t1 - t0
    llm_time += step_llm_time

    # ---- 工具执行时间 ----
    if tool_name in tool_registry:
        t2 = time.time()
        tool_result = tool_registry[tool_name](**tool_args)
        t3 = time.time()
        step_func_time = t3 - t2
        func_time += step_func_time
        print(f"[Tool Result] {tool_result}  (time: {step_func_time:.6f} s)")

        # 把工具结果加入消息历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "content": tool_result,
            "name": tool_name
        })
    else:
        print(f"[Error] Unknown tool: {tool_name}")
        break

# ====== 最终统计 ======
total_time = llm_time + func_time
print("\n=== Final Breakdown ===")
print(f"Total time: {total_time:.3f} s")
print(f"  - LLM inference time: {llm_time:.3f} s")
print(f"  - Function call time: {func_time:.3f} s")
print(f"Total tokens: {total_tokens}")
print(f"Function call tokens: {func_tokens}")
ratio = func_tokens / total_tokens if total_tokens > 0 else 0
print(f"Function call token ratio: {ratio:.2%}")
