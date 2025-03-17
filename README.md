
# About
Naive Impl of [ToCa](https://github.com/Shenyi-Z/ToCa.git) based on diffusers.

It may take around 9G more gpu-vram caching.

Around 20% to 100% speed up. This fluctuation may be due to my gpu was shared and might serving others when i was testing.

It was not an accurately tested and is for reference only.


# Installation & Usage

```bash
git clone https://github.com/wuutiing/diffusers_toca_flux.git
cd diffusers_toca_flux && git submodule update --init --recursive
bash _copy.sh # replace modification of diffusers
cd diffusers && python3 -m pip install .
cd ../ && python3 example.py 

```
