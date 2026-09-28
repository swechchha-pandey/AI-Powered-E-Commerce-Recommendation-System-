from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [
            req.strip()
            for req in requirements
            if req.strip() and not req.strip().startswith("-e")
        ]
    return requirements

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="product_recommendation_system",
    version="0.1.0",
    author="Samriddhi Sonker",
    author_email="sonkersamriddhi@gmail.com",
    description="A product recommendation system using browsing history",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # 🔥 THIS IS THE FIX
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=get_requirements("requirements.txt"),

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)