import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

max_count = 10
llm = init_chat_model(
    model="openai:qwen3.8-max",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    temperature=0,
)


# --------------tool--------------------------


@tool
def get_product_price(device: str) -> float:
    """Fetch price of the device，available device scope is laptop,Headset,keyboard
    args:
        device: device name
    returns: price of the device
    """
    print(f"    >>start the get_price tool with query={device}")
    device_scope = {"laptop": 1299, "headset": 120, "keyboard": 30}
    return device_scope[device]


@tool
def get_product_discount(price: float, medal_type) -> float:
    """Calculate the final price with medal_type
    args:
        price: price of the device
        medal_type: scope cover glod silver bronze
    returns: the final price with discount
    """
    print(
        f"    >>start the get_product_discount tool with price={price},medal={medal_type}"
    )
    model_discount_dict = {"bronze": 5, "silver": 10, "glod": 15}
    discount = round(1 - model_discount_dict[medal_type] / 100, 2)
    return discount


tools = [get_product_price, get_product_discount]
tool_dict = {t.name: t for t in tools}
model = llm.bind_tools(tools)


# --------------Agent loop--------------------------
def agent_loop():
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content="laptop多少钱"),
    ]

    for i in range(1, max_count + 1):
        ai_message = model.invoke(messages)

        tool_call = ai_message.tool_calls
        if tool_call is None:
            return ai_message.content

        for tool_call_item in tool_call:
            tool_name = tool_call_item.get("name", "")
            tool_arg = tool_call_item.get("args", {})
            tool_call_id = tool_call_item.get("id", {})

            if tool_name in tool_dict.keys():
                tool_to_use = tool_dict.get(tool_name)
                observation = tool_to_use.invoke(tool_arg)
                print(f"    >>the tool called {tool_name} with args={tool_arg}")
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    return None


agent_loop()
