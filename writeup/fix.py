import re
import sys
m = {"https://arxiv.org/abs/1409.0473": "DBLP:journals/corr/BahdanauCB14",
     "https://arxiv.org/abs/1607.06450": "ba2016layer",
     "https://arxiv.org/abs/1512.03385": "he2016deep",
     "https://arxiv.org/abs/1308.0850": "DBLP:journals/corr/Graves13",
     "http://jmlr.org/papers/v15/srivastava14a.html":"srivastava2014dropout",
     "https://arxiv.org/abs/1703.03906": "DBLP:journals/corr/BritzGLL17" ,
     "https://arxiv.org/abs/1608.05859": "DBLP:journals/corr/PressW16",
     "https://arxiv.org/pdf/1705.03122.pdf": "DBLP:journals/corr/GehringAGYD17",
     "https://arxiv.org/abs/1412.6980": "DBLP:journals/corr/KingmaB14",
     "https://arxiv.org/abs/1512.00567": "DBLP:journals/corr/SzegedyVISW15"
}

for l in sys.stdin:
    s = l.strip().split()
    s2 = []
    for w in s:
        if w.startswith("\\href"):
            url = w.split("{")[1].split("}")[0]
            extra = w.split("}", 2)
            if url in m:
                s2.append(("\\cite{%s}" % m[url]) +
                          (extra[-1] if len(extra) == 3 else ""))
            else:
                s2.append(w)
        else:
            s2.append(w)
    print(" ".join(s2))
