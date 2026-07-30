import numpy as np, wave, onnxruntime as ort, iree.runtime as ireert, os
S=__import__("os").environ.get("MOON_SCRATCH") or __import__("os").path.expanduser("~/moonshine-v2")
G=f"{S}/stream/golden"; os.makedirs(G,exist_ok=True)
L,H,HD,DIM,VOCAB=6,8,40,320,32768; BOS,EOS,W,MAX=1,2,64,96
wavp=f"{S}/.venv/lib/python3.12/site-packages/moonshine_voice/assets/beckett.wav"
wv=wave.open(wavp,"rb"); pcm=np.frombuffer(wv.readframes(wv.getnframes()),np.int16).astype(np.float32)/32768.0
def onnx(p): return ort.InferenceSession(f"{S}/tiny/float/{p}.onnx",providers=["CPUExecutionProvider"])
fe=onnx("frontend")
feats=fe.run(None,{"audio_chunk":pcm.reshape(1,-1),"sample_buffer":np.zeros((1,79),np.float32),"sample_len":np.zeros((1,),np.int64),
  "conv1_buffer":np.zeros((1,320,4),np.float32),"conv2_buffer":np.zeros((1,640,4),np.float32),"frame_count":np.zeros((1,),np.int64)})[0]
win=feats[:,:W,:].copy()
def load(p):
    c=ireert.SystemContext(config=ireert.Config("local-task")); c.add_vm_module(ireert.VmModule.copy_buffer(c.instance,open(p,"rb").read())); return c.modules.module["main"]
enc,adp,pre,wp=load(f"{S}/stream/v2_encoder.vmfb"),load(f"{S}/stream/v2_adapter.vmfb"),load(f"{S}/stream/pre_masked.vmfb"),load(f"{S}/stream/wp_masked.vmfb")
emb=np.fromfile(f"{S}/baked/decoder/dec_embed.weight.bin",np.float32).reshape(VOCAB,DIM)
c0=np.fromfile(f"{S}/wp/cos.bin",np.float32); s0=np.fromfile(f"{S}/wp/sin.bin",np.float32); invf=np.arctan2(s0[1::2],c0[0::2])
def rope(p):
    a=p*invf; c=np.empty(40,np.float32); s=np.empty(40,np.float32); c[0::2]=np.cos(a); c[1::2]=np.cos(a); s[0::2]=-np.sin(a); s[1::2]=np.sin(a); return c.reshape(1,40),s.reshape(1,40)
enc_o=np.asarray(enc(win)); mem64=np.asarray(adp(np.arange(W,dtype=np.int32).reshape(1,W),enc_o)).reshape(1,W,DIM)
mem96=np.concatenate([mem64,np.zeros((1,MAX-W,DIM),np.float32)],axis=1).astype(np.float32)
mask=np.concatenate([np.zeros(W,np.float32),np.full(MAX-W,-1e30,np.float32)]).reshape(1,1,1,MAX).astype(np.float32)
# ---- PREFILL golden ----
pe=emb[BOS].reshape(1,1,DIM).copy().astype(np.float32)
pe.tofile(f"{G}/pre_in0.bin"); mem96.copy().tofile(f"{G}/pre_in1.bin"); mask.copy().tofile(f"{G}/pre_in2.bin")
po=[np.asarray(x).astype(np.float32) for x in pre(pe,mem96.copy(),mask.copy())]
for i,x in enumerate(po): x.tofile(f"{G}/pre_out{i}.bin")
lg=po[24].reshape(-1); tok1=int(lg.argmax())
print("PREFILL outputs:",len(po),"tok1=",tok1)
sK=[po[4*l].copy() for l in range(L)]; sV=[po[4*l+1].copy() for l in range(L)]
cK=[po[4*l+2].copy() for l in range(L)]; cV=[po[4*l+3].copy() for l in range(L)]
# ---- WITH_PAST step-1 golden (past=1) ----
e=emb[tok1].reshape(1,1,DIM).copy().astype(np.float32); cq,sq=rope(1); cq=cq.astype(np.float32); sq=sq.astype(np.float32)
inp=[e,cq,sq]
for l in range(L):
    inp+=[sK[l],sV[l],cK[l],cV[l]]
    if l==0: inp.append(mask)
for i,x in enumerate(inp): np.asarray(x).astype(np.float32).tofile(f"{G}/wp_in{i}.bin")
wo=[np.asarray(x).astype(np.float32) for x in wp(*inp)]
for i,x in enumerate(wo): x.tofile(f"{G}/wp_out{i}.bin")
tok2=int(wo[12].reshape(-1).argmax())
print("WITHPAST outputs:",len(wo),"tok2=",tok2)
print("WROTE decoder goldens; tok1,tok2=",tok1,tok2)
