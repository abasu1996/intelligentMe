import random

class SimpleAI:
    def __init__(self):
        self.knowledge = {}

    def learn(self, question, answer):
        self.knowledge[question.lower()] = answer

    def respond(self, question):
        return self.knowledge.get(question.lower(), "I don't know the answer to that.")

def main():
    ai = SimpleAI()
    print("Welcome to SimpleAI! Teach me by typing: learn <question> <answer>")
    print("Ask me a question by typing: ask <question>")
    print("Type 'exit' to quit.")

    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.startswith("learn "):
            parts = user_input[6:].split(" ", 1)
            if len(parts) == 2:
                question, answer = parts
                ai.learn(question, answer)
                print("Learned!")
            else:
                print("Usage: learn <question> <answer>")
        elif user_input.startswith("ask "):
            question = user_input[4:]
            print(ai.respond(question))
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()