import numpy as np, wave, onnxruntime as ort, iree.runtime as ireert
S=__import__("os").environ.get("MOON_SCRATCH") or __import__("os").path.expanduser("~/moonshine-v2")
G=f"{S}/stream/golden"; import os; os.makedirs(G,exist_ok=True)
L,H,HD,DIM,VOCAB=6,8,40,320,32768; BOS=1; W=64
wavp=f"{S}/.venv/lib/python3.12/site-packages/moonshine_voice/assets/beckett.wav"
w=wave.open(wavp,"rb"); n=w.getnframes(); pcm=np.frombuffer(w.readframes(n),np.int16).astype(np.float32)/32768.0
audio=pcm.reshape(1,-1).astype(np.float32)
def onnx(p): return ort.InferenceSession(f"{S}/tiny/float/{p}.onnx",providers=["CPUExecutionProvider"])
fe=onnx("frontend")
fo=fe.run(None,{"audio_chunk":audio,"sample_buffer":np.zeros((1,79),np.float32),"sample_len":np.zeros((1,),np.int64),
  "conv1_buffer":np.zeros((1,320,4),np.float32),"conv2_buffer":np.zeros((1,640,4),np.float32),"frame_count":np.zeros((1,),np.int64)})
feats=fo[0]; win=feats[:, :W, :].copy().astype(np.float32)
def load(p):
    c=ireert.SystemContext(config=ireert.Config("local-task")); c.add_vm_module(ireert.VmModule.copy_buffer(c.instance, open(p,"rb").read())); return c.modules.module["main"]
enc,adp=load(f"{S}/stream/v2_encoder.vmfb"),load(f"{S}/stream/v2_adapter.vmfb")
pos=np.arange(W,dtype=np.int32).reshape(1,W)
enc_o=np.asarray(enc(win)).astype(np.float32)
mem_o=np.asarray(adp(pos, enc_o)).astype(np.float32)
win.tofile(f"{G}/enc_in.bin"); enc_o.tofile(f"{G}/enc_out.bin")
pos.tofile(f"{G}/adp_pos.bin"); enc_o.tofile(f"{G}/adp_in.bin"); mem_o.tofile(f"{G}/adp_out.bin")
print("enc_in",win.shape,"enc_out",enc_o.shape,"adp_out",mem_o.shape)
print("WROTE goldens to",G)
