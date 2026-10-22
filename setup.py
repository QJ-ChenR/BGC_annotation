from setuptools import setup, find_packages

setup(
    name="natural_product",  
    version="0.1",
    packages=find_packages(),  # 自动扫描所有含 __init__.py 的目录
    extras_require={
        "visualization": ["numpy", "pandas", "matplotlib", "umap-learn==0.5.12"],
        "dna": ["torch>=2.4", "transformers==5.1.0", "numpy", "biopython"],
    },
)
