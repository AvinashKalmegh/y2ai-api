from dotenv import load_dotenv
load_dotenv()

import os
print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_KEY:", os.getenv("SUPABASE_KEY"))
print("ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY"))
