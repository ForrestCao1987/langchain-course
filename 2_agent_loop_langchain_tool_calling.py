import json
import os

from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable
from openai import OpenAI

max_iterations = 10

# --- tool ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """look up the price of the product，available product： laptop,headphone,keyboard
    args:
        product: product name
    returns: price of the product
    """
    print(f"    >>Executing the get_product_price tool with query={product}")
    prices = {"laptop": 1299.99, "headphone": 149.95, "keyboard": 89.90}
    return prices.get(product, 0)


@traceable(run_type="tool")
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


@traceable(name="qwen_chat_traced", run_type="llm")
def qwen_chat_traced(messages, tools):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL")
    )
    response = client.chat.completions.create(
        model="qwen3.8-max",
        temperature=0,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let the model decide
    )
    return response.choices[0].message


# --- Agent loop ---
@traceable(name="Langchain Agent Loop")
def agent_loop(questions: str):

    # Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
    # This is exactly what LangChain's @tool decorator generates automatically
    # from the function's type hints and docstring.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_price",
                "description": "look up the price of the product，available product： laptop,headphone,keyboard",
                "parameters": {
                    "type": "object",
                    "required": ["product"],
                    "properties": {
                        "product": {"type": "string", "description": "product name"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "apply a discount tier to a product price and return the final price",
                "parameters": {
                    "type": "object",
                    "required": ["price", "discount_tier"],
                    "properties": {
                        "price": {
                            "type": "float",
                            "description": "price of the product",
                        },
                        "discount_tier": {
                            "type": "string",
                            "description": "available tiers: gold silver bronze",
                        },
                    },
                },
            },
        },
    ]

    # NOTE: Ollama can also auto-generate these schemas if you pass the functions
    # directly as tools (similar to LangChain's @tool decorator):
    #   tools_for_llm = [get_product_price, apply_discount]
    # However, this requires your docstrings to follow the Google docstring format
    # so Ollama can parse parameter descriptions from the Args section. For example:
    #   def get_product_price(product: str) -> float:
    #       """Look up the price of a product in the catalog.
    #
    #       Args:
    #           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
    #
    #       Returns:
    #           The price of the product, or 0 if not found.
    #       """
    # We keep the manual JSON version here so you can see what @tool hides from you.

    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {"role": "user", "content": questions},
    ]

    for i in range(1, max_iterations + 1):
        print(f"--- Iteration {i} ---")
        ai_message = qwen_chat_traced(messages, tools)

        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(ai_message.content)
            return None

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_arg = json.loads(tool_call.function.arguments)
            print(f"    >>[tool selected] with {tool_name} and arg {tool_arg}")

            if tool_name in tool_dict.keys():
                tool_to_use = tool_dict.get(tool_name)
                observation = tool_to_use(**tool_arg)
                print(f"    >>[tool result] {observation}")
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        messages.append({"role": "assistant", "content": str(ai_message)})
        messages.append(
            {"role": "tool", "content": str(observation), "name": tool_name}
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
