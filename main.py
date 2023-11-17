from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from process_audio import process_audio_stream

import io
import numpy as np

import base64

import matplotlib.pyplot as plt
import torch
from Zernike import mask_circle

N_pupil = 64
device = torch.device('cpu')

f = 400 # [m]
pixel_size = 10e-6 # [m]

λ = 1300e-9 # [m]
D = 8 # [m]
img_size = 64 # [pixels]
oversampling = 2 # []

pupil = torch.tensor( mask_circle(N_pupil, N_pupil//2)[None,...] ).to(device).float()

pixels_λ_D = f/pixel_size * λ/D
pad = np.round((oversampling*pixels_λ_D-1)*pupil.shape[-2]/2).astype('int')
φ_size = pupil.shape[-1] + 2*pad

padder = torch.nn.ZeroPad2d(pad.item())


def binning(inp, N):
    return torch.nn.functional.avg_pool2d(inp.unsqueeze(1),N,N).squeeze(1) * N**2 if N > 1 else inp

def OPD2PSF(λ, OPD, φ_size, padder, oversampling):  
    EMF = padder( pupil * torch.exp(2j*torch.pi/λ*OPD*1e-9) )

    lin = torch.linspace(0, φ_size-1, steps=φ_size, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing='xy')
    center_aligner = torch.exp(-1j*torch.pi/φ_size*(xx+yy)*(1-img_size%2))

    PSF = torch.fft.fftshift(1./φ_size * torch.fft.fft2(EMF*center_aligner, dim=(-2,-1)), dim=(-2,-1)).abs()**2
    cropper = slice(φ_size//2-(img_size*oversampling)//2, φ_size//2+round((img_size*oversampling+1e-6)/2))

    PSF = binning(PSF[...,cropper,cropper], oversampling)
    return PSF

def GetPSF(phase_cube):
    return OPD2PSF(λ, phase_cube, φ_size, padder, oversampling)
    

app = FastAPI()
frontend = Jinja2Templates(directory="frontend")


# Serve static files
@app.get("/{filename}")
async def static(request: Request, filename: str):
    return FileResponse(f"frontend/{filename}")

@app.get("/")
async def root(request: Request):
    return frontend.TemplateResponse("index.html.j2", {"request": request})

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    socket = True
    count = 0

    # load target wf from numpy array
    target_wf = np.load("./frames/493.npy")
    
    tmax = max(target_wf.flatten())
    tmin = min(target_wf.flatten())
    while socket:
        count += 1
        try:
            webm_io = await websocket.receive_bytes()

            # convert bytes to numpy array
            audio_data = np.frombuffer(webm_io, dtype=np.uint32)

            wf = process_audio_stream(audio_data, 1000)

            # save numpy array to file
            # np.save(f"./{count}.npy", wf)

            diff = target_wf - wf

            diff_2 = torch.from_numpy(diff[None, ...].astype(np.float32)).float() / 5e9 # this goes to PSF propagator

            PSF = GetPSF(diff_2).squeeze().cpu().numpy()

            ## show wavefront from voice
            my_stringIObytes = io.BytesIO()
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            # ax = fig.gca()
            ax[0].imshow(target_wf, vmin=tmin, vmax=tmax)
            ax[1].imshow(wf, vmin=tmin, vmax=tmax)
            ax[2].imshow(PSF)

            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_title("Input Atmospheric Disturbance")

            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_title("Applied Correction")

            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[2].set_title("Star Image")

            # fig.tight_layout()
            fig.savefig(my_stringIObytes, dpi=75, format='png')
            my_stringIObytes.seek(0)
            base64_encoded_image = base64.b64encode(my_stringIObytes.read()).decode("utf-8")
            plt.close()

            # send image array to frontend
            await websocket.send_json({"image": base64_encoded_image})


        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            print("main socket closed")
            socket = False            


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
