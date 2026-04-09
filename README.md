# Density-based Traffic Signal Control with ESP32

This project is originally forked from TraffIQ-PH for a commissioned work. This time, instead of using reinforcement learning based traffic signal control, we will use density i.e. the number of vehicles per lane. The road network is also different, the collinear lanes share the same traffic signal phase.

## Installation

This only works on Linux machines, I extensively tested this in Ubuntu 22.04. Please, install that operating system first in a GPU-enabled machine. Make sure to check the "Install third-party software for graphics and Wi-Fi hardware..." to avoid installing the GPU drivers separately. 

Once you have the operating system ready, install uv. Open a terminal and type the code below.

``` 
sudo apt install curl 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a folder in your home directory and clone the repository.

```
cd ~
mkdir project
cd project
sudo apt install git
git clone https://github.com/lukasdante/density-traffic-control.git
```

If git asks for your global username and email configuration. Provide what it says, make sure you have cloned the repository by running the `git clone` line again. Now that you have cloned it, create a virtual environment and install the necessary requirements.

```
uv venv env
source env/bin/activate
```

You should see that the terminal now has `(env)` before each line. Run the code below to install the requirements.

```
uv pip install -r requirements.txt
```

