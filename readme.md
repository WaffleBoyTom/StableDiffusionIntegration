This is an HDA and updated code that lets you create prompts with Houdini then fetch the output.
You still need to run SD through Conda

what the code esentially does is create a .json with your parameters, the .py then reads the json to create the images then once you click fetch_script it fetches the output and deletes the .json


HOW TO :

Set your parameters on the HDA then click Create Prompt

Open Conda and activate your environment with :  conda activate nameofyourenv

run the updated script with : python directory/optimized_txt2img_WBT.py (it should be pasted in your stable diffusion directory so probably pathSD/optimizedSD/optimized_txt2img_WBT.py

once that has finished processing, fetch the images with 'fetch_script'

As of right now, this only exists for the optimized version of the txt2img.py but if you're interested I can update the normal version as well


if you haven't got Conda or anything: read these

https://www.howtogeek.com/830179/how-to-run-stable-diffusion-on-your-pc-to-generate-ai-images/

Demo video here (subscribe for more "cool" stuff)  : 

https://youtu.be/qwvWNhVUwkU




Hit me up on Discord : WaffleboyTom#1929
