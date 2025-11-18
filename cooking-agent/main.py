"""
Cooking AI Agent - Interactive Console Application

This application uses Microsoft Agent Framework with GitHub Models to create
an intelligent cooking assistant that can:
- Search for recipes based on ingredients or cuisine
- Extract ingredients from recipe text
- Provide cooking tips and techniques
- Answer cooking-related questions

Author: GitHub Copilot
Date: November 18, 2025
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from tools import search_recipes, extract_ingredients, get_cooking_tips


# Load environment variables
load_dotenv()


def print_banner():
    """Display welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  🍳 COOKING AI AGENT 🍳                      ║
║                                                              ║
║  Your intelligent cooking assistant powered by AI            ║
║  - Recipe Search                                             ║
║  - Ingredient Extraction                                     ║
║  - Cooking Tips & Techniques                                 ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Type 'help' for usage examples or 'quit' to exit.\n")


def print_help():
    """Display help message with example queries"""
    help_text = """
📖 HOW TO USE:

Recipe Search:
  - "Find me pasta recipes"
  - "Show me vegan desserts"
  - "I want to make something with chicken"
  - "Search for gluten-free Indian recipes"

Ingredient Extraction:
  - "Extract ingredients from: [paste recipe text here]"
  - "What ingredients are in this recipe: [recipe text]"

Cooking Tips:
  - "Give me tips on knife skills"
  - "How do I meal prep effectively?"
  - "What are the food safety guidelines?"
  - "Tell me about baking techniques"

General Questions:
  - "How do I make carbonara?"
  - "What's the difference between baking and roasting?"
  - "How long should I cook chicken breast?"

Commands:
  - 'help' - Show this help message
  - 'clear' - Clear conversation history
  - 'quit' or 'exit' - Exit the application
    """
    print(help_text)


async def create_cooking_agent() -> ChatAgent:
    """
    Create and configure the cooking AI agent with GitHub Models
    
    Returns:
        ChatAgent: Configured cooking agent instance
    """
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        print("❌ ERROR: GITHUB_TOKEN not found in environment variables.")
        print("\nPlease follow these steps:")
        print("1. Create a .env file in the cooking-agent directory")
        print("2. Add your GitHub token: GITHUB_TOKEN=your_token_here")
        print("3. Get a token from: https://github.com/settings/tokens")
        print("\nFor more details, see the README.md file.")
        sys.exit(1)
    
    # Initialize OpenAI client with GitHub Models endpoint
    openai_client = AsyncOpenAI(
        base_url="https://models.github.ai/inference",
        api_key=github_token,
    )
    
    # Create chat client with gpt-4.1-mini model
    # This model offers great performance for cooking tasks at low cost
    chat_client = OpenAIChatClient(
        async_client=openai_client,
        model_id="openai/gpt-4.1-mini"
    )
    
    # Define agent instructions
    instructions = """You are an expert cooking assistant with deep knowledge of:
- International cuisines and recipes
- Cooking techniques and methods
- Food safety and nutrition
- Ingredient substitutions
- Meal planning and preparation

Your personality:
- Friendly, encouraging, and patient
- Enthusiastic about food and cooking
- Clear and practical in your explanations
- Creative with recipe suggestions

When users ask about recipes, use the search_recipes tool to find relevant options.
When users want to extract ingredients from text, use the extract_ingredients tool.
When users ask for cooking tips, use the get_cooking_tips tool.
For general cooking questions, provide helpful, accurate, and practical advice.

Always be conversational and engaging. Use emojis occasionally to make responses more friendly.
If a user seems to be a beginner, offer simpler recipes and more detailed explanations."""
    
    # Create the agent with tools
    agent = ChatAgent(
        chat_client=chat_client,
        name="CookingAgent",
        instructions=instructions,
        tools=[search_recipes, extract_ingredients, get_cooking_tips],
    )
    
    return agent


async def main():
    """Main application loop"""
    print_banner()
    
    print("🔧 Initializing Cooking AI Agent...")
    
    try:
        agent = await create_cooking_agent()
        print("✅ Agent ready! How can I help you with cooking today?\n")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Create a persistent thread for conversation context
    thread = agent.get_new_thread()
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Cooking AI Agent! Happy cooking!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                thread = agent.get_new_thread()
                print("🔄 Conversation history cleared.\n")
                continue
            
            # Process user query with the agent
            print("\n🤖 Agent: ", end="", flush=True)
            
            response_text = ""
            async for chunk in agent.run_stream(user_input, thread=thread):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_text += chunk.text
            
            print("\n")  # Add newline after response
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thanks for using Cooking AI Agent!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
