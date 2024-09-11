import asyncio
from functools import wraps
import os
from pathlib import Path
import threading
import uuid
from flask import Flask, request, jsonify, Response
from flask_basicauth import BasicAuth
from agent import AgentContext
from initialize import initialize
from python.helpers.files import get_abs_path
from python.helpers.print_style import PrintStyle
from python.helpers.log import Log
from dotenv import load_dotenv


#initialize the internal Flask server
app = Flask("app",static_folder=get_abs_path("./webui"),static_url_path="/")
lock = threading.Lock()

# Set up basic authentication, name and password from .env variables
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME') or "admin" #default name
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD') or "admin" #default pass
basic_auth = BasicAuth(app)

# get context to run agent zero in
def get_context(ctxid:str):
    with lock:
        if not ctxid: return AgentContext.first() or AgentContext(config=initialize())
        return AgentContext.get(ctxid) or AgentContext(config=initialize(),id=ctxid)

# Now you can use @requires_auth function decorator to require login on certain pages
def requires_auth(f):
    @wraps(f)
    async def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (auth.username == app.config['BASIC_AUTH_USERNAME'] and auth.password == app.config['BASIC_AUTH_PASSWORD']):
            return Response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return await f(*args, **kwargs)
    return decorated

# handle default address, show demo html page from ./test_form.html
@app.route('/', methods=['GET'])
async def test_form():
    return Path(get_abs_path("./webui/index.html")).read_text()

# simple health check, just return OK to see the server is running
@app.route('/ok', methods=['GET','POST'])
async def health_check():
    return "OK"

# # secret page, requires authentication
# @app.route('/secret', methods=['GET'])
# @requires_auth
# async def secret_page():
#     return Path("./secret_page.html").read_text()

# send message to agent (async UI)
@app.route('/msg', methods=['POST'])
async def handle_message_async():
    return await handle_message(False)

# send message to agent (synchronous API)
@app.route('/msg_sync', methods=['POST'])
async def handle_msg_sync():
    return await handle_message(True)

async def handle_message(sync:bool):
    try:
    
        #data sent to the server
        input = request.get_json()
        text = input.get("text", "")
        ctxid = input.get("context", "")
        blev = input.get("broadcast", 1)

        #context instance - get or create
        context = get_context(ctxid)

        # print to console and log
        PrintStyle(background_color="#6C3483", font_color="white", bold=True, padding=True).print(f"User message:")        
        PrintStyle(font_color="white", padding=False).print(f"> {text}")
        context.log.log(type="user", heading="User message", content=text)

        if sync:
            context.communicate(text)
            result = await context.process.result() #type: ignore       
            response = {
                "ok": True,
                "message": result,
            } 
        else:

            print("\n\n",(context.process and context.process.is_alive()))
            context.communicate(text)
            response = {
                "ok": True,
                "message": "Message received.",
            }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }

    #respond with json
    return jsonify(response)
        
# pausing/unpausing the agent
@app.route('/pause', methods=['POST'])
async def pause():
    try:
        
        #data sent to the server
        input = request.get_json()
        paused = input.get("paused", False)
        ctxid = input.get("context", "")

        #context instance - get or create
        context = get_context(ctxid)

        context.paused = paused

        response = {
            "ok": True,
            "message": "Agent paused." if paused else "Agent unpaused.",
            "pause": paused
        }        
        
    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }

    #respond with json
    return jsonify(response)

# restarting with new agent0
@app.route('/reset', methods=['POST'])
async def reset():
    try:

        #data sent to the server
        input = request.get_json()
        ctxid = input.get("context", "")

        #context instance - get or create
        context = get_context(ctxid)
        context.reset()
        
        response = {
            "ok": True,
            "message": "Agent restarted.",
        }        
        
    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }

    #respond with json
    return jsonify(response)

# killing context
@app.route('/remove', methods=['POST'])
async def remove():
    try:

        #data sent to the server
        input = request.get_json()
        ctxid = input.get("context", "")

        #context instance - get or create
        AgentContext.remove(ctxid)
        
        response = {
            "ok": True,
            "message": "Context removed.",
        }        
        
    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }

    #respond with json
    return jsonify(response)

# Web UI polling
@app.route('/poll', methods=['POST'])
async def poll():
    try:
        
        #data sent to the server
        input = request.get_json()
        ctxid = input.get("context", uuid.uuid4())
        from_no = input.get("log_from", 0)

        #context instance - get or create
        context = get_context(ctxid)


        logs = context.log.output(start=from_no)

        # loop AgentContext._contexts
        ctxs = []
        for ctx in AgentContext._contexts.values():
            ctxs.append({
                "id": ctx.id,
                "no": ctx.no,
                "log_guid": ctx.log.guid,
                "log_version": len(ctx.log.updates),
                "log_length": len(ctx.log.logs),
                "paused": ctx.paused
            })
        
        #data from this server    
        response = {
            "ok": True,
            "context": context.id,
            "contexts": ctxs,
            "logs": logs,
            "log_guid": context.log.guid,
            "log_version": len(context.log.updates),
            "paused": context.paused
        }

    except Exception as e:
        response = {
            "ok": False,
            "message": str(e),
        }

    #respond with json
    return jsonify(response)



#run the internal server
if __name__ == "__main__":

    load_dotenv()
    
    # Suppress only request logs but keep the startup messages
    from werkzeug.serving import WSGIRequestHandler
    class NoRequestLoggingWSGIRequestHandler(WSGIRequestHandler):
        def log_request(self, code='-', size='-'):
            pass  # Override to suppress request logging

    # run the server on port from .env
    port = int(os.environ.get("WEB_UI_PORT", 0)) or None
    app.run(request_handler=NoRequestLoggingWSGIRequestHandler,port=port)
