\# ROCm



\[https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html?fam=ryzen\&w=compute\&gpu=amd-ryzen-ai-max-pro-395\&os=windows\&windows-ver=11\&i=pip\&gfx=gfx1151#prerequisites]



\[https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.1.0\&fam=ryzen\&gpu=amd-ryzen-ai-max-pro-395\&os=windows\&pytorch-ver=2.14.0\&i=pip\&w=compute\&gfx=gfx1151]





    uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm\[libraries,device-gfx1151]==10.1.0"



    uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch\[device-gfx1151]==2.14.0+rocm10.1.0" "torchvision\[device-gfx1151]==0.29.0a0+rocm10.1.0" "torchaudio==2.11.0.3+rocm10.1.0"

flash-linear-attention
causal_conv1d

