import json
import time
from openai import OpenAI
import os

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
                    "unit": {"type": "string", "description": "Temperature unit", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

# 本地工具函数
def get_current_weather(city: str, state: str, unit: str):
    return f"The weather in {city}, {state} is 85 degrees {unit}. Partly cloudy, highs in the 90's."

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
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=0.95,
        max_tokens=1024,
        tools=tools,
    )
    t1 = time.time()
    step_llm_time = t1 - t0
    llm_time += step_llm_time

    choice = response.choices[0].message
    usage = response.usage
    total_tokens += usage.total_tokens

    print("\n=== Model Response ===")
    print(choice)

    # function call token 计入
    if choice.tool_calls:
        func_tokens += usage.completion_tokens

    # 如果没有工具调用，则说明模型给出最终回答
    if not choice.tool_calls:
        print("\n=== Final Answer ===")
        print(choice.content)
        break

    # ---- 工具执行时间 ----
    for tool_call in choice.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print(f"\n[Tool Call] {tool_name}({tool_args})")

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
                "tool_call_id": tool_call.id,
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