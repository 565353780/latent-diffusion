pip install -U torch torchvision torchaudio

pip install -U albumentations opencv-python pudb imageio imageio-ffmpeg \
	pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity \
	transformers

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
