<h1 align="center">
  Eric Chat
</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0" height="20">
  </a>

</p>

A Mac GUI for running LLMs locally and privately. Eric Chat uses [Eric Transformer](https://github.com/EricFillion/erictransformer) which is powered by Apple's [MLX-LM](https://github.com/ml-explore/mlx-lm) for fast and memory efficient inference.  


| Available Models                                                                    |
|-------------------------------------------------------------------------------------|
| [EricFillion/smollm3-3b-mlx](https://huggingface.co/EricFillion/smollm3-3b-mlx)     |                                                               
| [EricFillion/gpt-oss-20b-mlx](https://huggingface.co/EricFillion/gpt-oss-20b-mlx)   |                                                             
| [EricFillion/gpt-oss-120b-mlx](https://huggingface.co/EricFillion/gpt-oss-120b-mlx) |                                                            


## Install

macOS 14 or higher and Apple silicon are required. 

```sh
pip install ericchat
```

## Launch
### Terminal 
```sh
python3 -m ericchat
```

### Python
```python
from ericchat import run

run()
```


## Maintainers
- [Eric Fillion](https://github.com/ericfillion) Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer

## Contributing 
We are currently not accepting contributions. 
