((*- extends 'article.tplx' -*))

((* block docclass *))
\documentclass[11pt,a4paper]{article}

\usepackage[hyperref]{acl2018}

    \usepackage[T1]{fontenc}
    % Nicer default font (+ math font) than Computer Modern for most use cases
    \usepackage{times}
    \usepackage{latexsym}
\makeatletter

                                                                                              
    \title{The Annotated Transformer}
    \author{Alexander M. Rush \\ srush@seas.harvard.edu \\ Harvard University}
\usepackage{etoolbox}
\aclfinalcopy

\usepackage[indentfirst=true,leftmargin=0pt, rightmargin=0pt]{quoting}
\AtBeginEnvironment{quoting}{\fontfamily{phv}\selectfont}
((* endblock docclass *))



((* block abstract *))

\begin{abstract}
  A major aim of open-source NLP is to quickly and accurately
  reproduce the results of new work, in a manner that the community
  can easily use and modify. While most papers publish enough detail
  for replication, it still may be difficult to achieve good results
  in practice. This paper is an experiment. In it, I consider a worked
  exercise with the goal of implementing the results of the recent
  paper. The replication exercise aims at simple code structure that
  follows closely with the original work, while achieving an efficient
  usable system. An implicit premise of this exercise is to encourage
  researchers to consider this style as a method for releasing new
  results.
\end{abstract}

\section{Introduction}


Replication of published results remains a challenging issue
in open-source NLP. When a new paper is
published with major improvements, it is common for many
members of the community to independently  reproduce the 
numbers experimentally, which is often a struggle. Practically this makes it difficult to improve
scores, but more importantly it is a pedagogical issue if students
cannot reproduce results from scientific publications.

The recent turn towards deep learning has exerbated this issue. New
models require extensive hyperparameter tuning and long training
times. Small mistakes can cause major issues.  Fortunately though, new
toolsets have made it possible to write simpler more mathematically
declarative code.

In this experimental paper, I propose an exercise in open-source
NLP. The goal is to transcribe a recent paper into a simple and
understandable form. The document itself is presented as 
an annotated paper. That is the main document (in different font) is an excerpt of the recent paper ``Attention is All You Need''
\cite{DBLP:journals/corr/VaswaniSPUJGKP17}. I add annotation in the
form of italicized comments and include code in PyTorch directly in
the paper itself.

Note this document itself is presented as a blog post
\footnote{Presented at \url{http://nlp.seas.harvard.edu/2018/04/03/attention.html} with source code at \url{https://github.com/harvardnlp/annotated-transformer} } and is
completely executable as a notebook. In the spirit of reproducibility
this work itself is distilled from the same source with images
inline. 



((* endblock abstract *))

 ((* block postdoc *))
\section{Conclusion}


This paper presents a replication exercise of
  the transformer network. Consult the
  full online version for features such as multi-gpu
  training, real experiments on full translation problems, and
  pointers to other extensions such as beam search, sub-word models,
  and model averaging, necessary for state-of-the-art performance.

  The main goal of this paper is to experimenally  explore a literate programming exercise of
  interleaving model replication with formal writing. While not always
  possible, this modality can be useful for transmitting ideas and
  encouraging faster open-source uptake. Additionally this method can
  be an easy way to learn about a model alongside its implementation.

\bibliographystyle{acl}
\bibliography{ref}
((* block bibliography *))
((* endblock bibliography *))
((* endblock postdoc *))



((* block input scoped *))
    ((*- if cell.metadata.hide *))
    ((*- else -*))
    \begin{tiny}
        \noindent
    \begin{Verbatim}[commandchars=\\\{\}]
((( cell.source | highlight_code(strip_verbatim=True, metadata=cell.metadata) )))
    \end{Verbatim}
     \end{tiny}
     ((*- endif -*))
((* endblock input *))


((* block execute_result scoped *))
    ((*- if cell.metadata.hide_output *))
    ((*- else -*))
   \noindent
    \begin{tiny}
    ((*- for type in output.data | filter_data_type -*))
       ((*- if type in ['text/plain'] *))
              ((( output.data['text/plain'] )))
       ((*- else -*))
       ((* block data_priority scoped *))
       ((( super() )))
       ((* endblock -*))
       ((*- endif -*))
    ((*- endfor -*))
    \end{tiny}
  ((*- endif -*))
((* endblock execute_result *))


% Render markdown
((* block markdowncell scoped *))

    ((*- if cell.metadata.hide *))
    ((*- else -*))
    \begin{quoting}
((( cell.source | citation2latex | strip_files_prefix | convert_pandoc('markdown+tex_math_double_backslash', 'json',extra_args=[]) | resolve_references | convert_pandoc('json','latex'))))
     \end{quoting}
         ((*- endif -*))
 ((* endblock markdowncell *))
