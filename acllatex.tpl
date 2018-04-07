((*- extends 'article.tplx' -*))

((* block docclass *))
\documentclass[11pt,a4paper]{article}

\usepackage[hyperref]{acl2018}

    \usepackage[T1]{fontenc}
    % Nicer default font (+ math font) than Computer Modern for most use cases
    \usepackage{times}
    \usepackage{latexsym}

    \title{The Annotated Transformer}
    \author{Alexander M. Rush}


((* endblock docclass *))


((* block input scoped *))

    \begin{tiny}
        \noindent
    \begin{Verbatim}[commandchars=\\\{\}]
((( cell.source | highlight_code(strip_verbatim=True, metadata=cell.metadata) )))
    \end{Verbatim}
     \end{tiny}
((* endblock input *))


((* block execute_result scoped *))
    \noindent
    \begin{tiny}
    ((*- for type in output.data | filter_data_type -*))
       ((*- if type in ['text/plain'] *))
              ((( output.data['text/plain'] )))
       ((*- else -*))
              ((( super() )))
       ((*- endif -*))
    ((*- endfor -*))
    \end{tiny}
((* endblock execute_result *))
