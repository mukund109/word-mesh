import setuptools
import re
import io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('wordmesh/__init__.py', encoding='utf_8').read()
    ).group(1)

setuptools.setup(
    name="wordmesh",
    version=__version__,
    author="Mukund Chaudhry",
    author_email="mukund.chaudhry@gmail.com",
    description="A wordcloud generator which allows for meaningful word clustering",
    license = 'MIT',
    url="https://github.com/mukund109/word-mesh",
    install_requires=['textacy>=0.6.1', 'plotly>=2.0.11', 'numpy>=1.12.0', 'colorlover==0.2.1', 'pandas>=0.19.2', 'scipy>=0.18.1', 'scikit_learn'],
    python_requires='>=3',
    packages=['wordmesh'],
    package_data={'wordmesh': ['stopwords.txt']}

)
