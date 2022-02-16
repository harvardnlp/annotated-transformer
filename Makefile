notebook: the_annotated_transformer.py
	jupytext --to ipynb the_annotated_transformer.py

py: the_annotated_transformer.ipynb
	jupytext --to py:percent the_annotated_transformer.ipynb

the_annotated_transformer.ipynb: the_annotated_transformer.py
	jupytext --to ipynb the_annotated_transformer.py

execute: the_annotated_transformer.py
	jupytext --execute --to ipynb the_annotated_transformer.py

html: the_annotated_transformer.ipynb
	jupytext --to ipynb the_annotated_transformer.py
	jupyter nbconvert --to html the_annotated_transformer.ipynb

flake: the_annotated_transformer.ipynb
	flake8 --show-source the_annotated_transformer.py

black: the_annotated_transformer.ipynb
	black --line-length 79 the_annotated_transformer.py

clean: 
	rm -f the_annotated_transformer.ipynb
