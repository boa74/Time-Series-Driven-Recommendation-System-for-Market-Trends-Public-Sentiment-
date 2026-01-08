#!/bin/bash

cd "$(dirname "$0")"

echo "Stopping any existing Flask server on port 18502..."
lsof -ti:18502 | xargs kill -9 2>/dev/null

echo "Starting Flask server..."
export FLASK_APP=app.py
export FLASK_ENV=development

nohup python3 app.py > flask.log 2>&1 &
SERVER_PID=$!

echo "Flask server started with PID: $SERVER_PID"
echo "Server log: flask.log"
echo "Access the application at: http://127.0.0.1:18502"

sleep 2

if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Server is running successfully!"
else
    echo "✗ Server failed to start. Check flask.log for errors."
    cat flask.log
fi
