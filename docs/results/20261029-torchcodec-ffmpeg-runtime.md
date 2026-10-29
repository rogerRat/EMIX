# 2026-10-29 TorchCodec and FFmpeg runtime repair

Last modified: 2026-10-29 CST.

## Decision

Use TorchCodec for EMIX training video reads. Leave the video backend unset unless an experiment explicitly selects one, so LeRobot can choose the installed decoder. Do not force PyAV as a generic workaround for a broken TorchCodec runtime.

## Evidence

- Successful historical training snapshots did not record the video backend. The training config builder left it unset, and the emix environment had TorchCodec installed. This strongly suggests LeRobot selected TorchCodec for those runs. Historical evaluation fallback messages came from a separate evaluation process and do not establish the training backend.
- On 2026-10-29, four samples from the active EMIX dataset averaged 0.076 s/sample with TorchCodec and 2.512 s/sample with PyAV, about 33x slower for PyAV.
- The forced-PyAV run reached about 1,100/30,000 steps after roughly 1 h 45 min, with GPU utilization at 0% during long data-loader stalls. After switching back to TorchCodec, the restarted run passed 1,600 steps within about 3 min, GPU utilization was around 55%, and no decode exception appeared.

## Runtime repair

The host had FFmpeg 4 core libraries but lacked libavdevice.so.58 and libavfilter.so.7, which TorchCodec needed to load its FFmpeg 4 core. For the active run, the missing Ubuntu packages and dependencies were extracted into /root/autodl-tmp/ffmpeg4-runtime; an emix Conda activation hook adds those library directories to LD_LIBRARY_PATH. TorchCodec import and decode of a real batch video both passed. The project installer now installs the system FFmpeg runtime on Debian/Ubuntu and includes a synthetic-video decode smoke check.

The active run uses TorchCodec. Its generated train config explicitly records video_backend: torchcodec; the project runner no longer forces PyAV when no backend is configured.
