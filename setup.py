from setuptools import setup, find_packages

requirements = """
ray==1.6.0
env==0.1.0
rich==10.12.0
wandb==0.12.7
numpy==1.19.5
pettingzoo==1.11.2
matplotlib==3.5.0
torch==1.10.0
gym==0.21.0
scipy==1.7.1
pillow==8.3.2
pyglet==1.5.21
setuptools==58.0.4
scikit-image==0.18.3
dm-tree==0.1.6
pandas==1.3.3
opencv-python==4.5.4.58
torchvision==0.11.1
tqdm==4.62.3
"""


setup(
    name='mambrl',
    version='1.0',
    packages=find_packages(),
    license='MIT',
    url='https://gitlab.com/nicofirst1/MAMBRL',
    install_requires=requirements.split("\n")

)

