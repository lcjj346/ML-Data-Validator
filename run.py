"""Entry point - run the FastAPI server.

Serves the API and (when frontend/dist exists) the built React UI on one URL:
http://localhost:8000

Binds to 127.0.0.1 by design: this tool processes sensitive data offline, so
it should never be reachable from other machines on the network.
"""

import os

import uvicorn

if __name__ == "__main__":
    dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    if os.path.isdir(dist):
        print("ML Data Validator -> http://localhost:8000")
    else:
        print("API only -> http://localhost:8000/docs")
        print("(frontend/dist not built - run 'cd frontend && npm run build' for the single-URL app,")
        print(" or 'cd frontend && npm run dev' for the dev server on http://localhost:5173)")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
