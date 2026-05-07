import sherpa_onnx
import sounddevice as sd
import numpy as np
import queue
import time

## For vietnamese TTS testing
# MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\Zipformer-30M-RNNT-Streaming-6000h"
# SAMPLE_RATE = 16000
# CHUNK_MS = 16
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

# recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
#     encoder=f"{MODEL_DIR}/encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
#     decoder=f"{MODEL_DIR}/decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
#     joiner=f"{MODEL_DIR}/joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
#     tokens=f"{MODEL_DIR}/config.json",
#     sample_rate=SAMPLE_RATE,
#     num_threads=2,
# )


## For English model_1
# MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\models\sherpa-onnx-streaming-zipformer-en-2023-06-26"
# SAMPLE_RATE = 16000
# CHUNK_MS = 16
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

# recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
#     encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
#     decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
#     joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
#     tokens=f"{MODEL_DIR}/tokens.txt",
#     sample_rate=SAMPLE_RATE,
#     num_threads=2,
# )

## For English model_1
MODEL_DIR = r"D:\STUDY\At_school\asr_zipformer\models\sherpa-onnx-streaming-zipformer-en-2023-06-21"
SAMPLE_RATE = 16000
CHUNK_MS = 16
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1.int8.onnx",
    joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    sample_rate=SAMPLE_RATE,
    num_threads=2,
)

stream = recognizer.create_stream()
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

print("🎤 Start speaking (Ctrl+C to stop)...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=CHUNK_SIZE,
    dtype="float32",
    callback=audio_callback,
):
    try:
        last_text = ""
        while True:
            audio = audio_queue.get()
            audio = audio.squeeze()          # (N, 1) -> (N,)

            stream.accept_waveform(SAMPLE_RATE, audio.tolist())

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            result = recognizer.get_result(stream)
            if result != last_text:
                print("Full Text: ", result)
                last_text = result

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStop recording")

recognizer.decode_stream(stream)
print("FINAL:", recognizer.get_result(stream))
