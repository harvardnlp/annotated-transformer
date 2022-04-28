Code for The Annotated Transformer blog post:

http://nlp.seas.harvard.edu/annotated-transformer/

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/AnnotatedTransformer.ipynb)



# Package Dependencies

Use `requirements.txt` to install library dependencies with pip:

```
pip install -r requirements.txt
```

# torchtext 0.12 unable to download IWSLT2016 - temporary workaround

Unfortunately the current version of torchtext does not download the IWSLT2016 dataset due to [a bug reported here](https://github.com/pytorch/text/issues/1676).

For the time being, follow the instructions from the issue (reproduced here) to download the dataset manually.

1. Download the IWSLT2016 data manually from [this link](https://drive.google.com/file/d/1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8/view). If the download worked the file size should be close to 187.6 MB in size.

2. When opening the annotated transformer notebook, in the jupyter lab file browser, click the upload button to upload the 2016-01.tgz file downloaded in step 1.

3. Open a terminal and move `2016-01.tgz` from the notebook folder to `~/.torchtext/cache/IWSLT2016/` using `mv 2016-01.tgz ~/.torchtext/cache/IWSLT2016/`. This is the cache location where torchtext checks for the dataset if it has already been downloaded.

Once torchtext resolves the download problem we should be able to eliminate these steps, [track the issue](https://drive.google.com/file/d/1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8/view) for the current status of the issue resolution.


# Notebook Setup

The Annotated Transformer is created using [jupytext](https://github.com/mwouts/jupytext).

Regular notebooks pose problems for source control - cell outputs end up in the repo history and diffs between commits are difficult to examine. Using jupytext, there is a python script (`.py` file) that is automatically kept in sync with the notebook file by the jupytext plugin.

The python script is committed contains all the cell content and can be used to generate the notebook file. The python script is a regular python source file, markdown sections are included using a standard comment convention, and outputs are not saved. The notebook itself is treated as a build artifact and is not commited to the git repository.

Prior to using this repo, make sure jupytext is installed by following the [installation instructions here](https://github.com/mwouts/jupytext/blob/main/docs/install.md).

To produce the `.ipynb` notebook file using the markdown source, run (under the hood, the `notebook` build target simply runs `jupytext --to ipynb the_annotated_transformer.py`):

```
make notebook
```

To produce the html version of the notebook, run:

```
make html
```

`make html` is just a shortcut for for generating the notebook with `jupytext --to ipynb the_annotated_transformer.py` followed by using the jupyter nbconvert command to produce html using `jupyter nbconvert --to html the_annotated_transformer.ipynb`                             
 

# Formatting and Linting

To keep the code formatting clean, the annotated transformer git repo has a git action to check that the code conforms to [PEP8 coding standards](https://www.python.org/dev/peps/pep-0008/).

To make this easier, there are two `Makefile` build targets to run automatic code formatting with black and flake8.

Be sure to [install black](https://github.com/psf/black#installation) and [flake8](https://flake8.pycqa.org/en/latest/).

You can then run:

```
make black
```

(or alternatively manually call black `black --line-length 79 the_annotated_transformer.py`) to format code automatically using black and:

```
make flake
```

(or manually call flake8 `flake8 --show-source the_annotated_transformer.py) to check for PEP8 violations.

It's recommended to run these two commands and fix any flake8 errors that arise, when submitting a PR, otherwise the github actions CI will report an error.
