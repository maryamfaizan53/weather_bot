from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field
import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import json
from functools import wraps
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Check if the API keys are present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
# Configure API keys


class WeatherState(BaseModel):
    """State management for the weather agent"""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_interaction: Optional[datetime] = None
    saved_locations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_interaction = datetime.now()
    
    def get_recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.conversation_history[-limit:]
    
    def save_location(self, location: str, data: Dict[str, Any]):
        self.saved_locations[location] = data

def tool_decorator(func: Callable) -> Callable:
    """Decorator to register functions as tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.is_tool = True
    return wrapper

class WeatherAgent:
    def __init__(
        self,
        name: str = "WeatherWise",
        temperature: float = 0.7,
        model: str = "gpt-4"
    ):
        self.state = WeatherState()
        self.name = name
        self.model = model
        self.temperature = temperature
        self.tools = self._register_tools()
        self.setup_agent()
    
    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register available tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or coordinates"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or coordinates"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days for forecast (1-5)"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_air_quality",
                    "description": "Get air quality index for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name or coordinates"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    
    def setup_agent(self):
        """Initialize the agent with its core instructions and capabilities"""
        self.instructions = """You are WeatherWise, an expert weather assistant powered by OpenAI.
        
        YOUR EXPERTISE:
        - Real-time weather data and forecasts
        - Weather patterns and phenomena
        - Climate science and meteorology
        - Weather-related safety advice
        - Air quality information
        
        CAPABILITIES:
        - Get current weather conditions
        - Provide weather forecasts
        - Check air quality
        - Save and manage locations
        - Track user preferences
        
        STYLE:
        - Use clear, accessible language
        - Include interesting weather facts when relevant
        - Be enthusiastic about meteorology
        - Always consider user's saved locations and preferences
        """
    
    @tool_decorator
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for a location using OpenWeatherMap API"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": self.state.user_preferences.get("units", "metric")
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return {"error": f"Failed to get weather data: {response.status_code}"}
    
    @tool_decorator
    def get_weather_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for a location"""
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": self.state.user_preferences.get("units", "metric")
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "forecast": data["list"][:days],
                "city": data["city"]["name"]
            }
        else:
            return {"error": f"Failed to get forecast data: {response.status_code}"}
    
    @tool_decorator
    def get_air_quality(self, location: str) -> Dict[str, Any]:
        """Get air quality index for a location"""
        # First get coordinates
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": location,
            "limit": 1,
            "appid": OPENWEATHER_API_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            location_data = response.json()
            if location_data:
                lat, lon = location_data[0]["lat"], location_data[0]["lon"]
                
                # Get air quality data
                aq_url = f"http://api.openweathermap.org/data/2.5/air_pollution"
                aq_params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": OPENWEATHER_API_KEY
                }
                aq_response = requests.get(aq_url, params=aq_params)
                if aq_response.status_code == 200:
                    aq_data = aq_response.json()
                    return {
                        "aqi": aq_data["list"][0]["main"]["aqi"],
                        "components": aq_data["list"][0]["components"]
                    }
        return {"error": "Failed to get air quality data"}
    
    def process_message(self, message: str) -> str:
        # Add user message to history
        self.state.add_to_history("user", message)
        
        # Prepare context from state
        context = {
            "recent_history": self.state.get_recent_history(),
            "saved_locations": self.state.saved_locations,
            "user_preferences": self.state.user_preferences
        }
        
        # Create the system message with context
        system_message = {
            "role": "system",
            "content": f"{self.instructions}\n\nContext: {json.dumps(context, indent=2)}"
        }
        
        # Create the user message
        user_message = {
            "role": "user",
            "content": message
        }
        
        # Get response from OpenAI with function calling
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=self.temperature,
            functions=self.tools,
            function_call="auto"
        )
        
        # Process the response
        response_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            
            # Call the appropriate function
            if function_name == "get_current_weather":
                result = self.get_current_weather(**function_args)
            elif function_name == "get_weather_forecast":
                result = self.get_weather_forecast(**function_args)
            elif function_name == "get_air_quality":
                result = self.get_air_quality(**function_args)
            
            # Get final response with function result
            final_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    system_message,
                    user_message,
                    response_message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(result)
                    }
                ],
                temperature=self.temperature
            )
            
            response_text = final_response.choices[0].message.content
        else:
            response_text = response_message.content
        
        # Add response to history
        self.state.add_to_history("assistant", response_text)
        
        return response_text
    
    def save_location(self, location: str, data: Dict[str, Any]):
        """Save a location and its associated data"""
        self.state.save_location(location, data)
    
    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        self.state.user_preferences.update(preferences)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current state"""
        return {
            "conversation_count": len(self.state.conversation_history),
            "last_interaction": self.state.last_interaction.isoformat() if self.state.last_interaction else None,
            "saved_locations": list(self.state.saved_locations.keys()),
            "preferences": self.state.user_preferences
        }

# Example usage
if __name__ == "__main__":
    # Create the weather agent
    weather_agent = WeatherAgent()
    
    # Example of saving a location
    weather_agent.save_location("New York", {
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York"
    })
    
    # Example of updating preferences
    weather_agent.update_preferences({
        "units": "metric",
        "notifications": True
    }) 
