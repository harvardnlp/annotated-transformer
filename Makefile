notebook: The\ Annotated\ Transformer.md
	jupytext --to ipynb The\ Annotated\ Transformer.md

py: The\ Annotated\ Transformer.md
	jupytext --to py The\ Annotated\ Transformer.md

The\ Annotated\ Transformer.ipynb: The\ Annotated\ Transformer.md
	jupytext --to ipynb The\ Annotated\ Transformer.md

markdown: The\ Annotated\ Transformer.md
	jupytext --to markdown The\ Annotated\ Transformer.md

execute: The\ Annotated\ Transformer.md
	jupytext --execute --to ipynb The\ Annotated\ Transformer.md

html: The\ Annotated\ Transformer.ipynb
	jupyter nbconvert --to html The\ Annotated\ Transformer.ipynb

install-jupytext-pip:
	pip install jupytext --upgrade

install-jupytext-conda:
	conda install jupytext -c conda-forge

flake: The\ Annotated\ Transformer.ipynb
	jupyter nbconvert The\ Annotated\ Transformer.ipynb --to python
	flake8 --show-source --ignore "N801, E203, E266, E501, W503, F812, E741, N803, N802, N806, W391" The\ Annotated\ Transformer.py

clean: 
	rm -f The\ Annotated\ Transformer.py The\ Annotated\ Transformer.ipynb