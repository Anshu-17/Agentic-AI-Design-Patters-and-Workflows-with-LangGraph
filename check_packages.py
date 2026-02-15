#!/usr/bin/env python3
"""Check which packages from requirements.txt are installed."""

packages_to_check = [
    'langchain',
    'langchain_core',
    'langchain_community',
    'langchain_google_genai',
    'langgraph',
    'pydantic',
    'typing_extensions',
    'dotenv',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'langsmith',
    'tavily',
    'langchain_openai',
    'tiktoken',
    'requests',
    'httpx',
    'aiohttp',
]

print("=" * 60)
print("PACKAGE INSTALLATION STATUS")
print("=" * 60)

installed = []
not_installed = []

for package in packages_to_check:
    try:
        __import__(package)
        print(f"✓ {package:30} INSTALLED")
        installed.append(package)
    except ImportError:
        print(f"✗ {package:30} NOT INSTALLED")
        not_installed.append(package)

print("\n" + "=" * 60)
print(f"SUMMARY: {len(installed)} installed, {len(not_installed)} missing")
print("=" * 60)

if not_installed:
    print("\nMissing packages:")
    for pkg in not_installed:
        print(f"  - {pkg}")
