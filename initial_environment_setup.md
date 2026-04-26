## Cloning the Audio Craft GitHub Repo

If you are using Mac with Apple Silicon Chip, the best way is to clone this repository to the local machine and then make some changes to the files available in the repository.  
These changes are necessary as [xFormers](https://github.com/facebookresearch/xformers.git), which is used for improving the compute efficiency, is not compatible with Apple silicon chips like M1, M2, etc...  

**Follow the below steps for succesful installation of [Audiocraft](https://github.com/facebookresearch/audiocraft.git)**  

- Fork the repository of [Audiocraft](https://github.com/facebookresearch/audiocraft.git) GitHub repository  
- Clone it to your local machine using the terminal command below.   
- Once cloned, Make the following changes
  - In file `requirements.txt`, make the following changes
  - Comment out the `xformers` and `torchtext==0.16.0`, since it is only for audio
  - Comment out `torch==2.1.0` and add `torch`
  - Comment out `torchaudio>=2.0.0,<2.1.2` and add `torchaudio`
  - Comment out `torchvision==0.16.0` and add `torchvision` 
  - Comment our `av=11.0.0` and add `av`
  - In the file `audiocraft/modules/transformer.py`
    - comment out the xformers import
      becomes
  - Around line 177 hard code self.memory_efficient to be false
    - Line of code
      becomes
  - Around line 189 comment out the memory_efficient check
    - Line of code 
      becomes
  - Around line 372 there's a call to unbind change the model from ops to torch
    - Line of code 
      becomes

## Creating new virtual environment

### Python Installation for Apple Silicon:

#### Python 3.9 Installation (as recommended for [Audiocraft](https://github.com/facebookresearch/audiocraft.git) that offers [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) model)

```bash
brew install python@3.9
```

#### Verification of installed python in

```bash
/opt/homebrew/opt/python@3.9/bin/python3.9 -V
```

```bash
/opt/homebrew/opt/python@3.9/bin/python3.9 -c "import platform; print(platform.machine())"
```

```bash
file /opt/homebrew/opt/python@3.9/bin/python3.9
```

> Expected output:  

- platform.machine() prints arm64   
- file ... mentions arm64

### Virtual Environment Creation for Mac (Apple Silicon chip)

```bash
cd "path to directory"
```

```bash
/opt/homebrew/opt/python@3.9/bin/python3.9 -m venv musicgen3.9.2
```

```bash
source musicgen3.9.2/bin/activate
```

### Installing Libraries

During Library installations, there are specific library dependencies for audiocraft. Also, while follwoing audiocraft documentation, we might face issues while building wheels for ffmpeg and av. However, we have commented out certain libraries and also removed the version pin for some, to make transformers copmatible.  

```bash
pip install --upgrade pip

pip install setuptools wheel

brew install ffmpeg


#The below '.' represents the root directory of audio craft. If your audiocraft is different from where your current directory is, then you need to change directory path pointing to the location where the audiocraft is located. Eg: /Users/user_name/directory/sub_directory/...../audiocraft
pip install -e . 

#Install torchcodec required to read .mp3 file
pip install torchcodec
pip install --upgrade torchcodec


#Creating jupyter kernel to create the enviroment in VS Code. 
pip install ipykernel

python3 -m ipykernel install --user --name=musicgen3.9.2 --display-name "Python (venv_musicgen3.9.2)"

```

After the above steps are complete, you can open your IDE and select the appropriate kernel space