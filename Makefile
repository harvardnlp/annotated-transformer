notebook: The\ Annotated\ Transformer.md
	jupytext --to ipynb The\ Annotated\ Transformer.md

The\ Annotated\ Transformer.ipynb: The\ Annotated\ Transformer.md
	jupytext --to ipynb The\ Annotated\ Transformer.md

execute: The\ Annotated\ Transformer.md
	jupytext --execute --to ipynb The\ Annotated\ Transformer.md

html: The\ Annotated\ Transformer.ipynb
	jupyter nbconvert --to html The\ Annotated\ Transformer.ipynb
