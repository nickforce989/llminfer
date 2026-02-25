from setuptools import setup, find_packages

setup(
    name="llminfer",
    version="0.1.0",
    description="GPU-Efficient LLM Inference Engine with quantization, KV cache, batching, and benchmarking",
    author="Your Name",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "serve": [
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
            "prometheus-client>=0.20.0",
        ],
        "peft": ["peft>=0.12.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio", "black", "isort", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "llminfer=llminfer.cli:app",
        ],
    },
)
