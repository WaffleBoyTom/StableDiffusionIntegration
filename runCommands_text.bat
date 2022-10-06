@echo off

:: path to your miniconda

call C:\ProgramData\Miniconda3\Scripts\activate.bat

call conda activate ldm

:: path to your stable diffusion

cd C:/StableDiffusion/stable-diffusion-main/stable-diffusion-main

::path to your ldm environment and path to the script

C:\Users\USERNAME\.conda\envs\ldm\python.exe "C:\StableDiffusion\stable-diffusion-main\stable-diffusion-main\scripts\txt2img_WBT.py"

pause


