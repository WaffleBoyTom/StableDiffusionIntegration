This is an HDA and updated code that lets you create prompts with Houdini then fetch the output.
You still need to run SD through Conda

what the code esentially does is create a .json with your parameters, the .py then reads the json to create the images then once you click fetch_script it fetches the output and deletes the .json


HOW TO :

Set your parameters on the HDA then click Create Prompt

You need to edit the paths in the .bat files

You need to edit the path to the .bat file in the python module of the HDA in the runSD() function

You need to edit the config and DEFAULT_CKPT variables in the .py scripts

You need to have the keyboard module installed inside of Houdini. if you have pip installed then just do : hython -m pip install keyboard

If you don't have pip installed , the great Paul Ambrosiussen has a video on youtube explaining how you can install it


if you haven't got Conda or anything: read these

https://www.howtogeek.com/830179/how-to-run-stable-diffusion-on-your-pc-to-generate-ai-images/

Demo video here (subscribe for more "cool" stuff)  : 

https://youtu.be/sALAHsBlvy4


Hope this works for you and you can get something out of it !

Big thank you to everyone who's helped me with this : Dana Ericson on Facebook, Jenny from SideFx Support, Henry Foster over on the mops discord, Paul Ambrosiussen for the pip tutorial and Stack Overflow obviously


Hit me up on Discord : WaffleboyTom#1929
