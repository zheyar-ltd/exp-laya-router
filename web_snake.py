"""
Zheyar AI Labs - Laya Snake Visual Dashboard.
Symmetrical entry point for Snake Web Server.
"""

from web_dashboard import app

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Zheyar AI Labs - Laya Snake Web Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\nStarting Zheyar Laya Snake Web Server on http://localhost:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
