import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

max_iterations = 10

# --- tool ---


@tool
def get_product_price(product: str) -> float:
    """look up the price of the product，available device scope is laptop,headphone,keyboard
    args:
        product: product name
    returns: price of the product
    """
    print(f"    >>Executing the get_product_price tool with query={product}")
    prices = {"laptop": 1299.99, "headphone": 149.95, "keyboard": 89.90}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a product price and return the final price
    args:
        price: price of the product
        discount_tier: available tiers: gold silver bronze
    returns: the final price
    """
    print(
        f"    >>start the get_product_discount tool with price={price},medal={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    final_price = round(price * (1 - discount / 100), 2)
    return final_price


# --- Agent loop ---
@traceable(name="Langchain Agent Loop")
def agent_loop(questions: str):
    llm = init_chat_model(
        model="openai:qwen3.8-max",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0,
    )
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
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
        HumanMessage(content=questions),
    ]

    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i} ---")
        ai_message = model.invoke(messages)

        tool_call = ai_message.tool_calls
        if len(tool_call) == 0:
            print(ai_message.content)
            return None

        for tool_call_item in tool_call:
            tool_name = tool_call_item.get("name", "")
            tool_arg = tool_call_item.get("args", {})
            tool_call_id = tool_call_item.get("id", {})
            print(f"    >>[tool selected] with {tool_name} and arg {tool_arg}")

            if tool_name in tool_dict.keys():
                tool_to_use = tool_dict.get(tool_name)
                observation = tool_to_use.invoke(tool_arg)
                print(f"    >>[tool result] {observation}")
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print(f"ERROR: max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello langchain with (.bind agent tool)")
    print()
    print(
        agent_loop(
            "what is the price of the laptop after apply to the gold medal tier."
        )
    )
