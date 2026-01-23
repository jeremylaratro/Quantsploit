from setuptools import setup, find_packages

setup(
    name="quantsploit",
    version="0.2.0",
    description="Quantitative Analysis Trading Framework with Interactive TUI",
    author="Quantsploit Team",
    packages=find_packages(),
    install_requires=[
        "prompt-toolkit>=3.0.0",
        "rich>=13.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "yfinance>=0.2.0",
        "pandas-ta>=0.3.14b",
        "py_vollib>=1.0.1",
        "scipy>=1.10.0",
        "python-dateutil>=2.8.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        'console_scripts': [
            'quantsploit=quantsploit.main:main',
        ],
    },
    python_requires='>=3.8',
)
