import asyncio
import logging
import base64
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# App-specific imports
from app.config import settings
from app.lifecycle import initialize_global_engines, shutdown_global_engines
from app.ltm_buffer import start_ltm_worker, stop_ltm_worker
from app import globals as G

# Project-specific service imports
from connectivity import start_connectivity_monitoring, stop_connectivity_monitoring
from services.reminder_checker import start_reminder_checker, stop_reminder_checker
from tools.scheduler import init_db as init_scheduler_db
from utils.music_db import init_music_likes_table
from tools.music_control import trigger_mpd_library_update

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan: Startup phase beginning...")
    await start_connectivity_monitoring()
    try:
        await init_scheduler_db()
        await init_music_likes_table()
        logger.info("Attempting to trigger MPD library update on startup...")
        if not await trigger_mpd_library_update():
            logger.warning("MPD library update command failed to send on startup.")
    except Exception as e:
        logger.error(f"Failed to initialize databases or trigger MPD update: {e}.", exc_info=True)
        
    await initialize_global_engines()
    await start_ltm_worker(G.mem0_client)

    if G.tts_engine and G.audio_output_engine and await G.tts_engine.is_healthy() and G.audio_output_engine.is_enabled:
        if not settings.scheduler_db_path:
            logger.warning("Scheduler DB path not set. Reminder checker might not function correctly.")
        await start_reminder_checker(G.tts_engine, G.audio_output_engine)
    else:
        logger.warning("TTS or Audio Output engine not available/healthy, reminder checker not started.")

    if G.audio_input_engine and G.audio_input_engine.is_enabled and \
       G.audio_output_engine and G.audio_output_engine.is_enabled and \
       G.manager and not G.manager.is_websocket_active:
        logger.info("Lifespan: No active WebSocket, starting local audio I/O.")
        G.audio_output_engine.start()
        G.audio_input_engine.start()
        G.manager.state = "wakeword"
        logger.info("Lifespan: Local audio interface activated, waiting for wake word.")
    
    logger.info("Lifespan: Startup phase complete.")
    yield
    logger.info("Lifespan: Shutdown phase beginning...")
    await stop_ltm_worker()
    await stop_reminder_checker()
    await shutdown_global_engines()
    await stop_connectivity_monitoring()
    logger.info("Lifespan: Shutdown phase complete.")


app = FastAPI(
    title="Lumi Voice Assistant Backend (Modular)",
    lifespan=lifespan
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory at: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_root():
    static_file_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(static_file_path):
        logger.error(f"Frontend file not found at: {static_file_path}")
        raise HTTPException(status_code=404, detail=f"Frontend file 'index.html' not found in '{static_dir}'.")
    return FileResponse(static_file_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not G.manager:
        logger.error("ConnectionManager not initialized for WebSocket endpoint.")
        await websocket.accept()
        await websocket.close(code=1011, reason="Server not ready")
        return

    connected = await G.manager.connect(websocket)
    if not connected:
        logger.info("WebSocket connection attempt failed or was rejected by ConnectionManager.")
        return

    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64 and G.manager and G.manager.websocket is websocket and G.manager.is_websocket_active:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        await G.manager.process_audio(audio_bytes, G.ASSISTANT_THREAD_ID, is_local_source=False)
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                        await G.manager.send_error("Server error processing audio.")
            
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected by client: code={e.code}, reason='{e.reason}'")
    except Exception as e:
        logger.error(f"Unexpected WebSocket Error: {e}", exc_info=True)
    finally:
        if G.manager and G.manager.websocket is websocket:
            logger.info(f"Calling manager.disconnect for websocket {websocket.client}")
            await G.manager.disconnect(reason="WebSocket endpoint cleanup")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Lumi Voice Assistant server (Modular) on {settings.webapp.host}:{settings.webapp.port}")
    
    if settings.sounddevice.enabled:
        try:
            import sounddevice as sd
            logger.info("Available Audio Devices (for SoundDevice engine):")
            logger.info(f"\n{sd.query_devices()}")
            logger.info(f"Using Input Device Index: {settings.sounddevice.input_device_index if settings.sounddevice.input_device_index is not None else 'Default'}")
            logger.info(f"Using Output Device Index: {settings.sounddevice.output_device_index if settings.sounddevice.output_device_index is not None else 'Default'}")
            logger.info(f"SoundDevice TTS Output Sample Rate (default): {settings.sounddevice.tts_output_sample_rate} Hz")
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup: {e}")
    
    if settings.engines.tts_engine == "hybrid" or settings.engines.tts_engine == "gemini" or \
       settings.engines.tts_online_provider == "gemini" or settings.engines.tts_offline_provider == "gemini":
        logger.info(f"Gemini TTS configured with model: {settings.gemini_tts.model}, voice: {settings.gemini_tts.voice_name}, sample_rate: {settings.gemini_tts.sample_rate}Hz")
        if not settings.gemini_tts.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini TTS will not function.")
        else:
            logger.info("GEMINI_API_KEY is set.")

    if settings.audio.play_activation_sound:
        if settings.audio.activation_sound_path: # Path is resolved and checked in settings
            logger.info(f"Activation sound enabled and file found: {settings.audio.activation_sound_path}")
        else:
            logger.warning("Activation sound enabled but ACTIVATION_SOUND_PATH is not set or file not found.")
    else:
        logger.info("Activation sound is disabled.")

    uvicorn.run("main:app", host=settings.webapp.host, port=settings.webapp.port, reload=True)