import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "RecruitMind AI"
    VERSION: str = "1.0.0"
    
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    
    GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
    
    def validate_keys(self):
        """
        Runs on startup to ensure all required keys are present.
        If they are missing, it stops the app and warns you.
        """
        missing_keys = []
        
        if not self.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
            
        if not self.GITHUB_TOKEN:
            missing_keys.append("Warning: GITHUB_TOKEN is missing. Agent 02 will be limited.")
            
        if missing_keys:
            raise ValueError(f"🚨 Missing critical environment variables in .env file: {', '.join(missing_keys)}")

# Create a single instance of the settings to use throughout your app
settings = Settings()

# Run the validation immediately when this file is imported
settings.validate_keys()