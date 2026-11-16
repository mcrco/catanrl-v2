// catanrl v2 experiment log. Compile with:
//   typst compile report/report.typ report/report.pdf
// Macros live in macros.typ; each section in sections/ imports them.

#import "macros.typ": *

#set document(title: "catanrl v2: experiment log")
#set page(paper: "a4", margin: (x: 2.3cm, top: 2.3cm, bottom: 2.4cm), numbering: "1", number-align: center)
#set page(footer: context align(center, text(font: sans, size: 8.5pt, fill: muted, counter(page).display())))
#set text(font: serif, size: 10.8pt, lang: "en")
#set par(justify: true, leading: 0.66em, spacing: 1.05em)
// Numbers skip an empty level, so an experiment directly under a section reads 9.1, not 9.0.1.
#set heading(numbering: (..n) => n.pos().filter(x => x > 0).map(str).join("."))
#show heading: set text(font: sans, weight: "semibold")
#show heading.where(level: 1): it => { pagebreak(weak: true); set text(size: 17pt); it; v(0.4em) }
#show heading.where(level: 2): it => { v(0.6em); set text(size: 13pt); it; v(0.1em) }
#show heading.where(level: 3): set text(size: 11.5pt)
#show heading.where(level: 3): set block(above: 1.8em, below: 0.5em)
#show raw: set text(font: mono, size: 0.84em)
#show link: set text(fill: accent)
#show ref: set text(fill: accent)
#set list(indent: 0.6em, spacing: 0.6em)
#set enum(indent: 0.6em, spacing: 0.6em)
// Definition lists (/ Term: text) read as run-in paragraphs: "Term. text".
#set terms(separator: [. ], hanging-indent: 0pt, spacing: 1.05em)
#show figure: set block(above: 1.2em, below: 1.3em)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(font: sans, size: 9pt)
#show figure.caption: set align(left)
#show figure.caption: set par(justify: false, leading: 0.55em)
#show table: set text(font: sans, size: 9.4pt, number-width: "tabular")
#show table: set par(justify: false, leading: 0.5em)
#show table: it => { show raw: set text(size: 1.08em); it }
#set table(inset: (x: 5pt, y: 4pt))
#show outline: set text(font: sans, size: 9.5pt)
#show outline.entry.where(level: 1): set block(above: 0.8em)
#show outline.entry.where(level: 1): set text(weight: "semibold")

#align(center)[
  #v(1.5em)
  #text(font: sans, size: 26pt, weight: "bold")[catanrl v2]
  #v(-0.6em)
  #text(font: sans, size: 14pt, fill: muted)[Experiment log]
  #v(0.2em)
  #text(font: sans, size: 9.5pt, fill: muted)[Living document · last updated 3 November 2026 · #link("https://github.com/mcrco/catanrl-v2")[GitHub] · #link(wandb-project)[W&B]]
]
#v(1em)

#text(font: sans, size: 13pt, weight: "semibold")[Contents]
#outline(depth: 2, title: none)

#include "sections/front.typ"
#include "sections/arch.typ"
#include "sections/recipe.typ"
#include "sections/search.typ"
#include "sections/hidden.typ"
#include "sections/az.typ"
#include "sections/critic.typ"
#include "sections/canopy.typ"
#include "sections/playouts.typ"
#include "sections/generalist.typ"
#include "sections/trading.typ"
#include "sections/markov.typ"
#include "sections/deploy.typ"
#include "sections/llm.typ"
