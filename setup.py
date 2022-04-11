from setuptools import setup, find_packages

setup(
    author="Yunqi Yan",
    author_email="yyqdyx2009@live.com",
    description="Grid Resilience Assessment",
    url="https://git.tsinghua.edu.cn/yan-yq18",
    name="GridResilience",
    version="0.1",
    packages=find_packages('.'),
    install_requires=[],
    exclude_package_date={'': ['.gitignore'], '': ['dist'], '': 'build', '': 'utility.egg.info'},
)
