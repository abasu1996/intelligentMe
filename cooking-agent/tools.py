"""
Cooking Agent Tools

This module provides tools for the cooking AI agent to search for recipes
and extract ingredients from recipe text.
"""

from typing import Annotated


def search_recipes(
    query: Annotated[str, "The search query - can be an ingredient, cuisine type, or dish name"],
    diet: Annotated[str, "Optional dietary restriction: vegetarian, vegan, gluten-free, or any"] = "any"
) -> str:
    """
    Search for recipes based on ingredients, cuisine type, or dish name.
    Returns a list of recipe suggestions with brief descriptions.
    """
    # This is a mock implementation. In a production app, this would call
    # a recipe API like Spoonacular, Edamam, or TheMealDB
    
    recipe_database = {
        "pasta": [
            {
                "name": "Classic Spaghetti Carbonara",
                "cuisine": "Italian",
                "diet": "any",
                "description": "Creamy pasta with eggs, pancetta, and Parmesan cheese",
                "cook_time": "20 minutes",
                "difficulty": "Medium"
            },
            {
                "name": "Vegan Pasta Primavera",
                "cuisine": "Italian",
                "diet": "vegan",
                "description": "Fresh vegetables tossed with pasta in a light garlic sauce",
                "cook_time": "25 minutes",
                "difficulty": "Easy"
            },
            {
                "name": "Gluten-Free Penne Arrabbiata",
                "cuisine": "Italian",
                "diet": "gluten-free",
                "description": "Spicy tomato sauce with garlic and red chili flakes",
                "cook_time": "15 minutes",
                "difficulty": "Easy"
            }
        ],
        "chicken": [
            {
                "name": "Herb-Roasted Chicken",
                "cuisine": "American",
                "diet": "any",
                "description": "Whole chicken roasted with rosemary, thyme, and garlic",
                "cook_time": "1.5 hours",
                "difficulty": "Medium"
            },
            {
                "name": "Chicken Tikka Masala",
                "cuisine": "Indian",
                "diet": "gluten-free",
                "description": "Tender chicken in a creamy spiced tomato sauce",
                "cook_time": "45 minutes",
                "difficulty": "Medium"
            }
        ],
        "salad": [
            {
                "name": "Greek Salad",
                "cuisine": "Mediterranean",
                "diet": "vegetarian",
                "description": "Fresh vegetables with feta cheese, olives, and olive oil",
                "cook_time": "10 minutes",
                "difficulty": "Easy"
            },
            {
                "name": "Quinoa Buddha Bowl",
                "cuisine": "Modern",
                "diet": "vegan",
                "description": "Nutritious bowl with quinoa, roasted vegetables, and tahini dressing",
                "cook_time": "30 minutes",
                "difficulty": "Easy"
            }
        ],
        "indian": [
            {
                "name": "Vegetable Curry",
                "cuisine": "Indian",
                "diet": "vegan",
                "description": "Mixed vegetables in aromatic curry sauce with coconut milk",
                "cook_time": "35 minutes",
                "difficulty": "Medium"
            },
            {
                "name": "Butter Chicken",
                "cuisine": "Indian",
                "diet": "gluten-free",
                "description": "Creamy tomato-based curry with tender chicken pieces",
                "cook_time": "40 minutes",
                "difficulty": "Medium"
            }
        ],
        "dessert": [
            {
                "name": "Chocolate Lava Cake",
                "cuisine": "French",
                "diet": "vegetarian",
                "description": "Rich chocolate cake with a molten center",
                "cook_time": "25 minutes",
                "difficulty": "Hard"
            },
            {
                "name": "Vegan Banana Nice Cream",
                "cuisine": "Modern",
                "diet": "vegan",
                "description": "Creamy frozen dessert made from blended bananas",
                "cook_time": "5 minutes",
                "difficulty": "Easy"
            }
        ]
    }
    
    # Search logic
    query_lower = query.lower()
    results = []
    
    # Check if query matches any category
    for category, recipes in recipe_database.items():
        if category in query_lower or query_lower in category:
            results.extend(recipes)
    
    # If no category match, search in recipe names and descriptions
    if not results:
        for recipes in recipe_database.values():
            for recipe in recipes:
                if (query_lower in recipe["name"].lower() or 
                    query_lower in recipe["description"].lower() or
                    query_lower in recipe["cuisine"].lower()):
                    results.append(recipe)
    
    # Filter by diet if specified
    if diet != "any":
        results = [r for r in results if r["diet"] == diet or r["diet"] == "any"]
    
    # If still no results, return general suggestions
    if not results:
        return f"No specific recipes found for '{query}'. Try searching for: pasta, chicken, salad, indian, or dessert."
    
    # Format results
    output = f"Found {len(results)} recipe(s) for '{query}'"
    if diet != "any":
        output += f" with '{diet}' diet"
    output += ":\n\n"
    
    for i, recipe in enumerate(results[:5], 1):  # Limit to 5 results
        output += f"{i}. **{recipe['name']}** ({recipe['cuisine']} cuisine)\n"
        output += f"   - {recipe['description']}\n"
        output += f"   - Cooking time: {recipe['cook_time']} | Difficulty: {recipe['difficulty']}\n"
        output += f"   - Diet: {recipe['diet']}\n\n"
    
    return output


def extract_ingredients(
    recipe_text: Annotated[str, "The recipe text or description to extract ingredients from"]
) -> str:
    """
    Extract and list ingredients from a recipe text.
    Identifies common ingredients and their quantities.
    """
    # This is a simplified implementation. In production, you might use NLP
    # or a specialized ingredient extraction API
    
    common_ingredients = [
        # Proteins
        "chicken", "beef", "pork", "fish", "salmon", "shrimp", "tofu", "eggs",
        "turkey", "lamb", "bacon", "pancetta",
        
        # Vegetables
        "tomato", "onion", "garlic", "carrot", "celery", "potato", "bell pepper",
        "spinach", "broccoli", "mushroom", "zucchini", "eggplant", "lettuce",
        "cucumber", "pepper", "peppers",
        
        # Grains & Pasta
        "pasta", "rice", "quinoa", "bread", "flour", "noodles", "spaghetti",
        "penne", "couscous",
        
        # Dairy
        "cheese", "milk", "butter", "cream", "yogurt", "parmesan", "mozzarella",
        "feta", "cheddar",
        
        # Herbs & Spices
        "salt", "pepper", "oregano", "basil", "thyme", "rosemary", "cumin",
        "paprika", "cinnamon", "ginger", "curry", "chili", "garlic",
        
        # Oils & Sauces
        "olive oil", "oil", "soy sauce", "vinegar", "sauce", "stock", "broth",
        
        # Others
        "sugar", "honey", "lemon", "lime", "banana", "chocolate", "vanilla",
        "coconut milk", "chickpeas", "beans", "olives"
    ]
    
    # Units and measurements to identify
    units = ["cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", 
             "teaspoons", "tsp", "pound", "pounds", "lb", "lbs", "ounce", "ounces",
             "oz", "gram", "grams", "g", "kilogram", "kg", "ml", "liter", "l",
             "clove", "cloves", "piece", "pieces", "can", "cans"]
    
    text_lower = recipe_text.lower()
    found_ingredients = []
    
    # Extract ingredients with context
    words = text_lower.split()
    for i, word in enumerate(words):
        # Clean word of punctuation
        clean_word = word.strip(".,;:()[]{}\"'")
        
        # Check if word is an ingredient
        for ingredient in common_ingredients:
            if ingredient in clean_word or clean_word in ingredient:
                # Try to capture quantity (look at previous 1-3 words)
                context_start = max(0, i - 3)
                context = " ".join(words[context_start:i+1])
                
                # Check if there's a measurement
                has_measurement = any(unit in context for unit in units)
                
                if has_measurement:
                    found_ingredients.append(f"- {context}")
                else:
                    found_ingredients.append(f"- {ingredient}")
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ingredients = []
    for ing in found_ingredients:
        ing_lower = ing.lower()
        if ing_lower not in seen:
            seen.add(ing_lower)
            unique_ingredients.append(ing)
    
    if not unique_ingredients:
        return "No specific ingredients could be extracted. Please provide a more detailed recipe text with ingredient lists."
    
    output = f"Extracted {len(unique_ingredients)} ingredient(s):\n\n"
    output += "\n".join(unique_ingredients)
    output += "\n\nNote: This is an automated extraction. Please verify quantities and ingredients."
    
    return output


def get_cooking_tips(
    topic: Annotated[str, "The cooking topic to get tips about (e.g., 'knife skills', 'food safety', 'meal prep')"]
) -> str:
    """
    Provide cooking tips and techniques for various topics.
    """
    tips_database = {
        "knife skills": """
**Essential Knife Skills Tips:**

1. **Keep your knives sharp** - A sharp knife is safer than a dull one
2. **Use the claw grip** - Curl your fingers when holding food to protect fingertips
3. **Rock the knife** - Use a rocking motion for efficient chopping
4. **Choose the right knife** - Chef's knife for most tasks, paring knife for small work
5. **Practice your cuts** - Master dice, julienne, and chiffonade techniques
        """,
        
        "food safety": """
**Food Safety Guidelines:**

1. **Temperature control** - Keep cold foods below 40°F, hot foods above 140°F
2. **Wash hands frequently** - Before and after handling raw meat, eggs, or produce
3. **Separate raw and cooked** - Use different cutting boards for meat and vegetables
4. **Cook to proper temperature** - Use a meat thermometer to ensure safety
5. **Store properly** - Refrigerate perishables within 2 hours (1 hour if above 90°F)
        """,
        
        "meal prep": """
**Meal Prep Success Tips:**

1. **Plan your menu** - Choose recipes that store and reheat well
2. **Batch cook proteins** - Cook multiple portions of chicken, fish, or tofu
3. **Prep vegetables** - Wash, chop, and store in airtight containers
4. **Use proper containers** - Glass containers work best for reheating
5. **Label everything** - Date your containers and use within 3-4 days
        """,
        
        "seasoning": """
**Seasoning and Flavor Tips:**

1. **Salt in layers** - Season at different stages of cooking for depth
2. **Balance flavors** - Combine sweet, salty, sour, bitter, and umami
3. **Bloom spices** - Toast spices in oil to release their flavors
4. **Fresh herbs at the end** - Add delicate herbs just before serving
5. **Taste as you go** - Continuously adjust seasoning throughout cooking
        """,
        
        "baking": """
**Baking Fundamentals:**

1. **Measure accurately** - Use proper measuring tools for dry and liquid ingredients
2. **Room temperature ingredients** - Butter and eggs should be at room temp
3. **Don't overmix** - Overmixing develops gluten and makes baked goods tough
4. **Preheat your oven** - Always preheat for at least 15-20 minutes
5. **Rotate pans** - Rotate halfway through for even baking
        """
    }
    
    topic_lower = topic.lower()
    
    # Search for matching tips
    for key, tips in tips_database.items():
        if key in topic_lower or topic_lower in key:
            return tips
    
    # If no match, return general tips
    return f"""
No specific tips found for '{topic}'. Try asking about:
- Knife skills
- Food safety
- Meal prep
- Seasoning
- Baking

Or ask me any cooking question and I'll do my best to help!
    """
