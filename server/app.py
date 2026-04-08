"""
meeting_env/server/app.py
FastAPI server — wraps MeetingEnvironment and exposes OpenEnv HTTP endpoints.
"""
from openenv.core.env_server import create_fastapi_app
from .models import MeetingAction, MeetingObservation
from .meeting_environment import MeetingEnvironment

# Pass the class (not an instance) — the SDK instantiates the environment itself
app = create_fastapi_app(MeetingEnvironment, MeetingAction, MeetingObservation)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
