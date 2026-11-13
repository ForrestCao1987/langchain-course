import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("api_key"))


if __name__ == "__main__":
    main()
