# References — verification ledger

Method: an independent reference-checker agent received `refs.bib` only, resolved every DOI through the Crossref REST API (with OpenAlex / Semantic Scholar / dblp / publisher landing pages / arXiv / PubMed as second sources where Crossref was incomplete or the publisher page returned 403), compared title, authors, journal, volume, pages/article number and year field by field, and read the abstract of every entry. Result: **41 of 41 DOIs resolve; 2 author-list errors found and corrected** (Yamilov 2023: fifth author is Zongfu **Yu**, not "Liu"; Imagawa 2010: **Shigeki** Imagawa and **Keisuke** Morita). One optional field added (Winkelmann 2019: `number = {2}`). No entry was dropped; nothing is cited that could not be resolved. Numerical-method sources marked † were additionally read in full or in key sections during the 2026-08-17 literature survey (`plans/2026-08-17_interior_literature_survey.md`, access disclosed there).

Legend for "what was read": CR = Crossref metadata record; abs = abstract (source in parentheses); full = full text read (survey).

| key | DOI | resolved | fields | what was read | cited for |
|---|---|---|---|---|---|
| Sellers2017 | 10.1038/ncomms14439 | yes | all OK | CR + abs (CR) | LSU networks, N/2 band rule, reference structure |
| Edagawa2008 | 10.1103/PhysRevLett.100.013901 | yes | all OK | CR + abs (OpenAlex) | amorphous-diamond 3-D gap |
| Imagawa2010 | 10.1103/PhysRevB.82.115116 | yes | authors corrected (Shigeki; Keisuke) | CR + abs (OpenAlex/APS) | PAD gap, diffusion, band-edge localized states |
| Florescu2009 | 10.1073/pnas.0907744106 | yes | all OK | CR + abs (CR) | hyperuniform designer gaps |
| Man2013 | 10.1073/pnas.1307879110 | yes | all OK | CR + abs (CR) | hyperuniform isotropic gap (experiment) |
| Liew2011 | 10.1103/PhysRevA.84.063818 | yes | all OK | CR + abs (OpenAlex) | 3-D networks with short-range order |
| Klatt2019 | 10.1073/pnas.1912730116 | yes | all OK | CR + abs (CR) | foam-based 3-D gaps; ε\|E\|² network-mode convention |
| Muller2017 | 10.1364/OPTICA.4.000361 | yes | OK (pages 361–366 confirmed via citing source) | CR + abs (OpenAlex/arXiv) | silicon hyperuniform networks |
| Muller2014 | 10.1002/adom.201300415 | yes | OK (vol 2, 115–119, 2014 print) | CR + abs (OpenAlex) | silicon hyperuniform SWIR gap |
| FroufePerez2017 | 10.1073/pnas.1705130114 | yes | all OK | CR + abs (CR) | gap formation ↔ Anderson localization with correlations |
| Haberko2020 | 10.1038/s41467-020-18571-w | yes | all OK | CR + abs (CR) | diffusion→localization near band edge, 3-D networks |
| Anderson1958 | 10.1103/PhysRev.109.1492 | yes | all OK | CR + abs (OpenAlex) | Anderson localization |
| Abrahams1979 | 10.1103/PhysRevLett.42.673 | yes | all OK | CR + abs (OpenAlex) | scaling theory |
| John1984 | 10.1103/PhysRevLett.53.2169 | yes | all OK | CR + abs (OpenAlex) | photon mobility edge |
| John1987 | 10.1103/PhysRevLett.58.2486 | yes | all OK | CR + abs (OpenAlex) | localization in pseudogap of dielectric superlattices |
| Wiersma1997 | 10.1038/37757 | yes | all OK | CR + abs (Nature page) | early 3-D localization report (later disputed) |
| Skipetrov2014 | 10.1103/PhysRevLett.112.023905 | yes | all OK | CR + abs (OpenAlex) | absence for vector waves / point scatterers |
| Sperling2016 | 10.1088/1367-2630/18/1/013039 | yes | OK (title quotes typographic) | CR + abs (IOP/OpenAlex) | white-paint claims withdrawn |
| SkipetrovPage2016 | 10.1088/1367-2630/18/2/021001 | yes | OK (Perspective confirmed) | CR + abs (IOP) | status of 3-D light localization |
| Yamilov2023 | 10.1038/s41567-023-02091-7 | yes | authors corrected (Yu, Zongfu) | CR + abs (Nature/S2) | numerical 3-D localization for metallic spheres, not dielectric |
| Oganesyan2007 | 10.1103/PhysRevB.75.155111 | yes | all OK | CR + abs (OpenAlex) | adjacent-gap ratio |
| Atas2013 | 10.1103/PhysRevLett.110.084101 | yes | all OK | CR + abs (OpenAlex) | ⟨r⟩ surmises and values |
| Shklovskii1993 | 10.1103/PhysRevB.47.11487 | yes | all OK | CR + abs (OpenAlex) | level statistics near the MIT |
| Evers2008 | 10.1103/RevModPhys.80.1355 | yes | all OK | CR + abs (OpenAlex) | Anderson transitions review |
| Johnson2001 † | 10.1364/OE.8.000173 | yes | OK (173–190) | CR + abs; MPB docs read (survey) | MPB formulation, preconditioner |
| Knyazev2001 † | 10.1137/S1064827500366124 | yes | all OK | CR + abs (CR) | LOBPCG |
| Duersch2018 | 10.1137/17M1129830 | yes | OK (C655–C676) | CR + abs (CR) | robust LOBPCG |
| Zhou2006 † | 10.1016/j.jcp.2006.03.017 | yes | all OK | CR + abs (snippet; survey abstract-level) | Chebyshev filtered subspace iteration |
| Weisse2006 † | 10.1103/RevModPhys.78.275 | yes | all OK | CR + abs (OpenAlex) | KPM, Jackson kernel |
| Pieper2016 † | 10.1016/j.jcp.2016.08.027 | yes | all OK | CR + abs (arXiv); Secs. 1–3 full (survey) | ChebFD interior filtering, degree scaling |
| Li2019 † | 10.1137/18M1170935 | yes | OK (C393–C415) | CR + abs; Maxwell sections full (survey) | EVSL, polynomial filtering for curl-curl |
| Winkelmann2019 † | 10.1145/3313828 | yes | OK (vol 45 no 2 art 21; `number` added) | CR + abs; arXiv (survey) | ChASE |
| Polizzi2009 † | 10.1103/PhysRevB.79.115112 | yes | all OK | CR + abs (OpenAlex) | FEAST (considered, not run) |
| Lin2016 † | 10.1137/130934283 | yes | all OK | CR + abs (CR) | spectral density estimation |
| Wang1994 † | 10.1063/1.466486 | yes | all OK | CR + abs; full PDF (survey) | folded spectrum |
| Vomel2008 † | 10.1016/j.jcp.2008.01.018 | yes | OK (7113–7124; full author list) | CR + abs (snippet); companion PARA'08 read in full (survey) | interior eigensolvers for nanostructures; shift-invert verdict |
| Farjadpour2006 | 10.1364/OL.31.002972 | yes | OK (2972–2974) | CR + abs (Optica) | subpixel smoothing (not applied) |
| Kottke2008 | 10.1103/PhysRevE.77.036611 | yes | all OK | CR + abs (OpenAlex) | anisotropic interface smoothing |
| FangSaad2012 † | 10.1137/110836535 | yes | all OK | CR + abs; full PDF (survey) | filtered Lanczos, near-miss failure mode |
| Lu2013 | 10.1038/nphoton.2013.42 | yes | OK (published title) | CR + abs (S2/arXiv) | gyroid crystal band indices |
| Lifshitz1964 | 10.1080/00018736400101061 | yes | OK (13, 483–536) | CR + abs (OpenAlex editor summary) | Lifshitz tails |
| SM | — (misc) | n/a | — | — | Supplemental Material pointer (PRL style) |

Not cited (from the required list): none. Every item of the kickoff's literature list is present and resolved. Sources cited only through the reference list of the Supplemental Material pointer (`SM` note): FangSaad2012, Lu2013 — both are also cited in the SM body.
