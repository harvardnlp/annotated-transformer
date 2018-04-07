((*- extends 'article.tplx' -*))

((* block docclass *))
\documentclass[11pt,a4paper]{article}

\usepackage[hyperref]{acl2018}

    \usepackage[T1]{fontenc}
    % Nicer default font (+ math font) than Computer Modern for most use cases
    \usepackage{times}
    \usepackage{latexsym}
((* endblock docclass *))

((* block input_group *))
(( cell.source ))
((* endblock input_group *))
