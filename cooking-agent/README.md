# 🍳 Cooking AI Agent

An intelligent cooking assistant powered by Microsoft Agent Framework and GitHub Models. This interactive console application helps you discover recipes, extract ingredients, and learn cooking techniques through natural conversation.

## ✨ Features

- **🔍 Recipe Search**: Find recipes based on ingredients, cuisine type, or dish name
- **📝 Ingredient Extraction**: Automatically extract ingredients from recipe text
- **💡 Cooking Tips**: Get expert advice on knife skills, food safety, meal prep, and more
- **💬 Interactive Chat**: Natural conversation interface for all your cooking questions
- **🧠 Context-Aware**: Remembers conversation history for follow-up questions
- **🚀 Powered by AI**: Uses GitHub Models (gpt-4.1-mini) for intelligent responses

## 🛠️ Tech Stack

- **Microsoft Agent Framework**: Flexible framework for building AI agents
- **GitHub Models**: Free-tier access to GPT-4.1-mini
- **Python**: Async/await for responsive interactions
- **OpenAI SDK**: For seamless model integration

## 📋 Prerequisites

- Python 3.9 or higher
- A GitHub Personal Access Token (PAT) - [Get one here](https://github.com/settings/tokens)
  - No specific scopes required for GitHub Models

## 🚀 Getting Started

### 1. Installation

Navigate to the cooking-agent directory and install dependencies:

```bash
cd cooking-agent

# Install dependencies
# NOTE: --pre flag is REQUIRED while Agent Framework is in preview
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the `cooking-agent` directory:

```bash
cp .env.example .env
```

Edit the `.env` file and add your GitHub token:

```env
GITHUB_TOKEN=your_github_token_here
```

**To get your GitHub token:**
1. Go to https://github.com/settings/tokens
2. Click "Generate new token" (classic)
3. No specific scopes are needed for GitHub Models
4. Copy the token and paste it in your `.env` file

### 3. Run the Application

```bash
python main.py
```

## 📖 Usage Examples

Once the application starts, you can interact with the cooking agent using natural language:

### Recipe Search
```
You: Find me pasta recipes
You: Show me vegan desserts
You: I want to make something with chicken
You: Search for gluten-free Indian recipes
```

### Ingredient Extraction
```
You: Extract ingredients from: To make carbonara, you need 400g spaghetti, 200g pancetta, 4 eggs, and 100g parmesan cheese
You: What ingredients are in this recipe: Mix flour, sugar, eggs, and butter to make cookies
```

### Cooking Tips
```
You: Give me tips on knife skills
You: How do I meal prep effectively?
You: What are the food safety guidelines?
You: Tell me about baking techniques
```

### General Questions
```
You: How do I make carbonara?
You: What's the difference between baking and roasting?
You: How long should I cook chicken breast?
You: Can I substitute butter with oil?
```

### Commands
- `help` - Display usage examples and commands
- `clear` - Clear conversation history and start fresh
- `quit` or `exit` - Exit the application

## 🏗️ Project Structure

```
cooking-agent/
├── main.py              # Main application entry point
├── tools.py             # Agent tools (recipe search, ingredient extraction, tips)
├── requirements.txt     # Python dependencies
├── .env.example        # Example environment configuration
├── .env               # Your configuration (create this)
└── README.md          # This file
```

## 🧩 How It Works

1. **Agent Framework**: The app uses Microsoft Agent Framework to create a conversational AI agent
2. **GitHub Models**: Connects to GitHub's free-tier models endpoint using your PAT
3. **Model Selection**: Uses `gpt-4.1-mini` for cost-effective, high-quality responses
4. **Tools**: The agent has access to three specialized tools:
   - `search_recipes()`: Searches a recipe database
   - `extract_ingredients()`: Parses text to find ingredients
   - `get_cooking_tips()`: Provides expert cooking advice
5. **Context**: Maintains conversation history using threads for natural follow-up questions

## 🎯 Why These Choices?

### GitHub Models
- **Free to start**: No charges until you hit rate limits
- **Easy setup**: Just need a GitHub PAT
- **Model variety**: Access to multiple models through one endpoint

### gpt-4.1-mini
- **Cost-effective**: Lower cost than full GPT-4
- **Fast**: Quick response times for interactive chat
- **Capable**: Excellent for cooking domain with quality ~0.81

### Microsoft Agent Framework
- **Tool support**: Easy integration of custom functions
- **Streaming**: Real-time response generation
- **Thread management**: Built-in conversation context
- **Flexible**: Works with multiple LLM providers

## 🔧 Customization

### Add More Recipes
Edit `tools.py` and expand the `recipe_database` dictionary in the `search_recipes()` function.

### Add More Tips
Edit `tools.py` and add entries to the `tips_database` dictionary in the `get_cooking_tips()` function.

### Change the Model
In `main.py`, modify the `model_id` parameter:
```python
chat_client = OpenAIChatClient(
    async_client=openai_client,
    model_id="openai/gpt-4.1"  # or another GitHub model
)
```

### Customize Agent Personality
Edit the `instructions` variable in `main.py` to change how the agent responds.

## 🐛 Troubleshooting

### "GITHUB_TOKEN not found"
- Make sure you created a `.env` file in the `cooking-agent` directory
- Verify the token is correctly set: `GITHUB_TOKEN=your_token_here`
- No quotes needed around the token value

### "Module not found" errors
- Ensure you installed dependencies with the `--pre` flag:
  ```bash
  pip install -r requirements.txt
  ```
- The `--pre` flag is required for the Agent Framework preview

### Rate limiting
- GitHub Models has free-tier rate limits
- If you hit limits, wait a few minutes or upgrade your GitHub account
- Consider switching to Azure AI Foundry for production use

## 📚 Learn More

- [Microsoft Agent Framework Documentation](https://github.com/microsoft/agent-framework)
- [GitHub Models](https://github.com/marketplace/models)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## 🤝 Contributing

This is a sample application. Feel free to:
- Add more recipe databases
- Integrate real recipe APIs (Spoonacular, Edamam, TheMealDB)
- Implement NLP for better ingredient extraction
- Add meal planning features
- Create a web interface

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Built with Microsoft Agent Framework
- Powered by GitHub Models
- Created with GitHub Copilot

---

**Happy Cooking! 🍳👨‍🍳👩‍🍳**
