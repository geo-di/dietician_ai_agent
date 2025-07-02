# === Environment and Dependencies ===
from dotenv import load_dotenv
import os
import random
import requests
import tkinter as tk
from tkinter import messagebox
from typing import Optional
from pydantic import BaseModel

# Langchain & Mistral
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# === Load Environment Variables ===
load_dotenv()

# === Data Models ===

class MealModel(BaseModel):
    name: str
    calories: int
    protein: float
    carbs: float
    fat: float
    ingredients: list[str]

class DailyMealPlanModel(BaseModel):
    breakfast: MealModel
    mid_morning_snack: MealModel
    lunch: MealModel
    afternoon_snack: MealModel
    dinner: MealModel
    total_calories: int
    macros: dict

class WeeklyMealPlanModel(BaseModel):
    day_1: DailyMealPlanModel
    day_2: DailyMealPlanModel
    day_3: DailyMealPlanModel
    day_4: DailyMealPlanModel
    day_5: DailyMealPlanModel
    day_6: DailyMealPlanModel
    day_7: DailyMealPlanModel
    note: Optional[str]

# === Tool Functions ===

# 1. Search Online for recipes or meal ideas
search = DuckDuckGoSearchRun()

# 2. Fetch a recipe based on ingredients and dietary preferences using Spoonacular API
def fetch_recipe(ingredients):
    base_url = "https://api.spoonacular.com/recipes/complexSearch"
    
    query = {
        "includeIngredients": ",".join(ingredients),
        "addRecipeInformation": True,
        "number": 5,
        "apiKey": os.getenv("SPOONACULAR_API_KEY")
    }

    if not query["apiKey"]:
        return {"error": "Missing Spoonacular API key."}

    response = requests.get(base_url, params=query)
    data = response.json()

    if "results" not in data or not data["results"]:
        return {"error": "No recipes found for the given ingredients and dietary preferences."}

    recipe = random.choice(data["results"])

    return {
        "name": recipe.get("title"),
        "calories": next((n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", []) if n["name"] == "Calories"), None),
        "protein": next((n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", []) if n["name"] == "Protein"), None),
        "carbs": next((n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", []) if n["name"] == "Carbohydrates"), None),
        "fat": next((n["amount"] for n in recipe.get("nutrition", {}).get("nutrients", []) if n["name"] == "Fat"), None),
        "ingredients": [i["name"] for i in recipe.get("extendedIngredients", [])],
    }

# 3. Check if meal plan's total calories match the target
def check_calories(meal_plan, calorie_target):
    total_calories = sum([meal['calories'] for meal in meal_plan.values()])
    return total_calories == calorie_target

# 4. Calculate macronutrients for a meal
def calculate_macros_for_meal(meal):
    return {
        'protein': meal['protein'],
        'carbs': meal['carbs'],
        'fat': meal['fat']
    }

# 5. Generate shopping list from a day's meal plan
def generate_shopping_list(day_plan_dict):
    shopping_list = []
    for meal_key in ['breakfast', 'lunch', 'dinner', 'mid_morning_snack', 'afternoon_snack']:
        meal = day_plan_dict.get(meal_key)
        if meal and 'ingredients' in meal:
            shopping_list.extend(meal['ingredients'])
    return list(set(shopping_list))  # Remove duplicates

# === Tools for the Agent ===
tools = [
    Tool(name="search_recipes_online", func=search.run, description="Search for recipes or meal ideas."),
    Tool(name="fetch_recipe", func=fetch_recipe, description="Fetch a recipe based on ingredients and dietary preferences."),
    Tool(name="check_calories", func=check_calories, description="Check calorie count of a given meal."),
    Tool(name="calculate_macros_for_meal", func=calculate_macros_for_meal, description="Calculate macronutrients for a meal."),
    Tool(name="generate_shopping_list", func=generate_shopping_list, description="Generate a shopping list for a week's meal plan.")
]

# === LLM and Agent Configuration ===
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-small-latest"
llm = ChatMistralAI(api_key=api_key, model_name=model, temperature=0.2)
parser = PydanticOutputParser(pydantic_object=WeeklyMealPlanModel)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a personal chef and meal planner. Generate a weekly meal plan with:
            - 5 meals/day: Breakfast, Mid-morning Snack, Lunch, Afternoon Snack, Dinner.
            - Include variety, balance, and respond to calorie/macro/diet requests.
            - If recipe fetching fails, use knowledge or search.
            - Add a note: portions may vary, consult a dietician.
            {format_instructions}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# === Meal Plan Generator ===

def get_meal_plan(calories, special_ingredients=None):
    user_query = f"Give me a 7-day meal plan with {calories} calories per day, high in protein, and low in carbs."
    if special_ingredients:
        user_query += f" Include these ingredients: {special_ingredients}."

    raw_response = agent_executor.invoke({"query": user_query})

    try:
        return parser.parse(raw_response["output"])
    except Exception as e:
        print("Failed to parse structured output:", e)
        return raw_response["output"]  # fallback to raw output

def generate_meal_output(weekly_meal_plan):
    output = ""
    for day in range(1, 8):
        day_plan = getattr(weekly_meal_plan, f"day_{day}", None)
        if not day_plan:
            continue

        output += f"Day {day}:\n"
        for meal_type in ['breakfast', 'mid_morning_snack', 'lunch', 'afternoon_snack', 'dinner']:
            meal = getattr(day_plan, meal_type)
            output += f"  {meal_type.replace('_', ' ').capitalize()}: {meal.name}\n"
            output += f"    Ingredients: {', '.join(meal.ingredients)}\n"
            output += f"    Calories: {meal.calories}, Protein: {meal.protein}g, Carbs: {meal.carbs}g, Fat: {meal.fat}g\n"

        output += f"  Total Calories: {day_plan.total_calories}\n"
        output += f"  Macros: Protein: {day_plan.macros['protein']}g, Carbs: {day_plan.macros['carbs']}g, Fat: {day_plan.macros['fat']}g\n"
        output += f"  Shopping List: {', '.join(generate_shopping_list(day_plan.dict()))}\n\n"

    output += f"Note: {getattr(weekly_meal_plan, 'note', '')}"
    return output

# === GUI Interface ===

def launch_gui():
    def on_submit():
        calories = calorie_entry.get()
        if not calories.isdigit():
            messagebox.showerror("Input Error", "Please enter a valid number for calories.")
            return

        special_ingredients = ingredients_entry.get()
        try:
            weekly_meal_plan = get_meal_plan(int(calories), special_ingredients)
            output_text.delete("1.0", tk.END)
            if isinstance(weekly_meal_plan, str):  # fallback case
                output_text.insert(tk.END, weekly_meal_plan)
            else:
                output_text.insert(tk.END, generate_meal_output(weekly_meal_plan))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Weekly Meal Planner")

    # User Inputs
    tk.Label(root, text="Enter Daily Calorie Intake:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    calorie_entry = tk.Entry(root)
    calorie_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Special Ingredients (optional):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    ingredients_entry = tk.Entry(root)
    ingredients_entry.grid(row=1, column=1, padx=10, pady=5)

    # Submit Button
    submit_button = tk.Button(root, text="Generate Meal Plan", command=on_submit)
    submit_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Output Text Box
    output_text = tk.Text(root, wrap="word", width=100, height=30)
    output_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

    root.mainloop()

# === Run GUI ===
if __name__ == "__main__":
    launch_gui()
