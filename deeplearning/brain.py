import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class SimpleAI:
    def __init__(self):
        self.knowledge = {}

    def learn(self, question, answer):
        self.knowledge[question.lower()] = answer

    def respond(self, question):
        return self.knowledge.get(question.lower(), "I don't know the answer to that.")

def main():
    # ai = SimpleAI()
    # print("Welcome to SimpleAI! Teach me by typing: learn <question> <answer>")
    # print("Ask me a question by typing: ask <question>")
    # print("Type 'exit' to quit.")

    # while True:
    #     user_input = input(">> ").strip()
    #     if user_input.lower() == "exit":
    #         break
    #     elif user_input.startswith("learn "):
    #         parts = user_input[6:].split(" ", 1)
    #         if len(parts) == 2:
    #             question, answer = parts

    #             print(f"Learning: {question} -> {answer}")
    #             ai.learn(question, answer)
    #             print("Learned!")
    #         else:
    #             print("Usage: learn <question> <answer>")
    #     elif user_input.startswith("ask "):
    #         question = user_input[4:]
    #         print(ai.respond(question))
    #     else:
    #         print("Unknown command.")

    df = pd.read_csv('car-sales.csv')
    print(df.head())
    print(df.describe())
    dtypes = df.dtypes 
    print(dtypes)
    df['Price'] = df['Price'].str.replace('[$,]', '', regex=True).astype(float)
    print(f'{df.columns}')
    ##Describe works only on numeric columns
    print(df.describe())
    df.head()
    df.head(10)
    print(df.head(10))
    ### iloc and .loc

    # iloc: integer-location based indexing
    # print(df.iloc[0])          # First row
    # print(df.iloc[0:3])        # First three rows

    # loc: label-based indexing
    # print(df.loc[0])           # Row with index label 0
    # print(df.loc[0:3])         # Rows with index labels 0 to 3 (inclusive)

    # # Selecting specific columns
    # print(df.loc[0, 'Make'])   # Value in row 0, column 'Make'
    # print(df.iloc[0, 1])       # Value in first row, second column

    # # Selecting multiple columns
    # print(df.loc[0:2, ['Make', 'Price']])  # Rows 0-2, columns 'Make' and 'Price'

    # df[df['Make'] == 'Toyota'].head()  # Filter rows where 'Make' is 'Toyota'


    # print(pd.crosstab(df['Make'], df['Doors']))
    print(df.groupby('Make').mean(numeric_only=True))
    missing_data = pd.read_csv('car-sales-missing-data.csv')
    print(missing_data.dropna())
    missing_data["Odometer"] = missing_data['Odometer'].fillna(missing_data["Odometer"].mean())

    print(f"MISSING FILL DATA {missing_data}")

if __name__ == "__main__":
    main()