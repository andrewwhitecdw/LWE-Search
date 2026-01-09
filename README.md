# LWE-Search 

LWE-Search is a tool that includes CUDA-accelerated LWE and Ring-LWE primitives and includes PIR (Private Information Retrieval) examples that leverage the aforementioned functionalities. 

## Build

Install the [CUDAToolkit](https://developer.nvidia.com/cuda-toolkit) if necessary, and then run the following:

```
git clone https://github.com/nvlabs/LWE-Search
cd LWE-Search
cmake -B build && cmake --build build -j$(nproc)
```

This will build all of the tests and examples; the generated binaries are located in the ``build/bin`` directory. 

## Contributors

LWE-Search originated as a project of [NVIDIA Research](https://research.nvidia.com).

Contributors to the initial open-source release (alphabetical): Chaz Gouert, Jongmin Kim, Jean-Luc Watson
