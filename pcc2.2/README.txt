
Potsdam Commentary Corpus 2.2
=============================

The Potsdam Commentary Corpus 2.2 (PCC 2.2) is a revised and extended version
of the Potsdam Commentary Corpus (Stede 2004), a collection of 176 German
newspaper commentaries (op-ed pieces) that has been annotated with syntax trees
and three layers of discourse-level information: nominal coreference,
connectives and their arguments (similar to the PDTB, Prasad et al. 2008), and
trees reflecting discourse structure according to Rhetorical Structure Theory
(Mann/Thompson 1988).

Connectives have been annotated with the help of a semi-automatic tool, Connanno
(Stede/Heintze 2004), which identifies most connectives and suggests arguments
based on their syntactic category. The other layers have been created manually
with dedicated annotation tools.


License
-------

The Potsdam Commentary Corpus 2.2 is released under a Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. You can find a
human-readable summary of the licence agreement here:

http://creativecommons.org/licenses/by-nc-sa/4.0/

If you are using our corpus for research purposes, please cite the following
paper:

Bourgonje, P and Stede, M (2020). The Potsdam Commentary Corpus 2.2: 
Extending Annotations for Shallow Discourse Parsing. Proc. of the Language 
Resources and Evaluation Conference (LREC), Marseille.

@inproceedings{bourgonje-stede-lrec2020,
    author = "Bourgonje, Peter and Stede, Manfred",
    title = "The Potsdam Commentary Corpus 2.2: Extending Annotations for Shallow Discourse Parsing",
    booktitle = "Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020) (to appear)",
    year = "2020",
    date = "11-16",
    month = "May",
    location = "Marseille, France",
    publisher = "European Language Resources Association (ELRA)",
    address = "Paris, France",
    keywords = ""
}



Corpus Directory Layout
-----------------------
.
├── connectives		coherence relations following the PDTB2 (Prasad et al. 2008) definition in standoff XML format
│
├── coreference		coreference, annotated with MMAX2 (Müller and Strube 2006)
│   │                   in MMAX2 XML format
│   ├── basedata
│   ├── customization
│   ├── markables
│   ├── schemes
│   └── styles
│
├── metadata		metadata for each document (author, title, publication
│                       date, document ID)
│
├── primary-data	original, untokenized documents in plain text UTF-8
│
├── rst			rhetorical structure, annotated with RSTTool
│                       (O'Donnell 2000) in RS3 format
│
├── syntax		sentence syntax following the Tiger scheme (Brants et al. 2004)
│                       in TigerXML format
│
└── tokenized		tokenized documents in plain text UTF-8



Version History
---------------
2.2 (2020-20-02)
~~~~~~~~~~~~~~~~~~

Version 2.2 extends/updates the PCC 2.1 as follows:
- The layer of connectives and their arguments, first introduced in the 2.0 version, is extended
by annotating them for senses, in line the with PDTB3.0 sense hierarchy.
- The additional relation types (taken from the PDTB2.0) of 'implicit', 'AltLex', 'EntRel' and 'NoRel'
have been annotated.
- Due to new annotations for implicits, the Conano XML format has disappeared from this release
and only the standoff XML format is available.
- Minor inconsistencies and errors from earlier versions have been corrected.

2.1.0 (2018-08-13)
~~~~~~~~~~~~~~~~~~

Version 2.1 extends/updates PCC 2.0 as follows:
- Aboutness topic: a new layer added to the corpus. Annotated with
Exmaralda. See: M. Stede / S. Mamprin: Information structure in the
Potsdam Commentary Corpus: Topics. In: Proc. of Language Resources and Evaluation Conference (LREC), Portoroz, 2016. 
- RST: maz14813.rs3 updated (old version was ill-formed)
- Syntax: maz17953.xml updated  (old version was ill-formed)
- connectives: various annotation errors corrected
- ParZu parses: automatically-produced dependency parses using ParZu
(see https://github.com/rsennrich/ParZu) and clevertagger (see
https://github.com/rsennrich/clevertagger), both developed at the
University of Zurich. The parses were made available by Don Tuggener.  
- coreference
--- coreference-mmax: same as the data provided in PCC 2.0
--- coreference-conll: This version was produced by Don Tuggener
(Zurich). Singleton markables have been removed, and the data is
converted to conll format
--- connectives-standoff: a standoff version of the inline XML format that conanno outputs. The annotations in this and connectives (inline format) are identical, the only thing different is the XML format.
For more details on the 2.1 version, see:
Peter Bourgonje and Manfred Stede. The Potsdam Commentary Corpus 2.1 in ANNIS3. 
In Proceedings of the 17th International Workshop on Treebanks and Linguistic Theory. 
Oslo, Norway, 2018

2.0.0 (2014-06-24)
~~~~~~~~~~~~~~~~~~

* release contains the PCC 2.0 corpus as described in Stede and Neumann (2014)
* annotation layers: syntax, rhetorical structure, coreference as well as
  connectives and their arguments


Bibliography
------------

Brants, S., Dipper, S., Eisenberg, P., Hansen, S., König, E., Lezius, W.,
Rohrer, C., Smith, G., and Uszkoreit, H. (2004).
TIGER: Linguistic interpretation of a German corpus.
Research on Language and Computation, 2(4):597–620.

Müller, C. and Strube, M. (2006). Multi-level annotation of linguistic data
with MMAX2. In Braun, S., Kohn, K., and Mukherjee, J., editors, 
Corpus Technology and Language Pedagogy: New Resources, New Tools, New Methods,
pages 197–214. Peter Lang, Frankfurt.

O’Donnell, M. (2000). RSTTool 2.4 – a markup tool for Rhetorical Structure
Theory. In Proceedings of the International Natural Language Generation
Conference, pages 253–256, Mizpe Ramon/Israel.

Stede, M. (2004). The Potsdam Commentary Corpus. In Proceedings of the ACL
Workshop on Discourse Annotation, pages 96–102.
Association for Computational Linguistics.

Stede, M. and Heintze, S. (2004). Machine-assisted rhetorical structure
annotation. In Proc. of the 20th International Conference on Computational
Linguistics, pages 425–431, Geneva.

Stede, M. and Neumann, A. (2014). Potsdam Commentary Corpus 2.0:
Annotation for Discourse Research. Proc. of the Language Resources and
Evaluation Conference (LREC), Reykjavik. 
