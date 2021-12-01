notebook: The\ Annotated\ Transformer.py
	jupytext --to ipynb The\ Annotated\ Transformer.py

The\ Annotated\ Transformer.ipynb: The\ Annotated\ Transformer.py
	jupytext --to ipynb The\ Annotated\ Transformer.py

execute: The\ Annotated\ Transformer.py
	jupytext --execute --to ipynb The\ Annotated\ Transformer.py

html: The\ Annotated\ Transformer.ipynb
	jupytext --to ipynb The\ Annotated\ Transformer.py
	jupyter nbconvert --to html The\ Annotated\ Transformer.ipynb

flake: The\ Annotated\ Transformer.ipynb
	flake8 --show-source The\ Annotated\ Transformer.py

black: The\ Annotated\ Transformer.ipynb
	black --line-length 79 The\ Annotated\ Transformer.py

clean: 
	rm -f The\ Annotated\ Transformer.ipynb
