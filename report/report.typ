// catanrl v2 experiment log. compile with:
//   typst compile report/report.typ report/report.pdf
// macros live in macros.typ; each section in sections/ imports them.

#import "macros.typ": *

#set document(title: "catanrl v2: experiment log")
#set page(paper: "a4", margin: (x: 2.1cm, top: 2.2cm, bottom: 2.3cm), numbering: "1", number-align: center)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em, spacing: 0.95em)
// numbers skip an empty level, so an experiment directly under a section reads 9.1, not 9.0.1
#set heading(numbering: (..n) => n.pos().filter(x => x > 0).map(str).join("."))
#show heading.where(level: 1): it => { v(0.9em); set text(size: 13.5pt); it; v(0.25em) }
#show heading.where(level: 2): it => { v(0.45em); set text(size: 11.2pt); it }
#show heading.where(level: 3): set text(size: 10.8pt)
#show heading.where(level: 3): set block(above: 1.5em, below: 0.4em)
#show raw: set text(font: "DejaVu Sans Mono", size: 0.86em)
#show link: set text(fill: accent)
#show ref: set text(fill: accent)
#set list(indent: 0.6em, spacing: 0.55em)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9pt)
#show figure.caption: set align(left)
#show table: set text(size: 8.8pt)
#show table: set par(justify: false)
#set table(stroke: none, inset: (x: 4.5pt, y: 2.9pt))
#show outline: set text(size: 9pt)
#show outline: set par(leading: 0.45em)
#show outline.entry.where(level: 1): set block(above: 0.55em)

#align(center)[
  #text(size: 21pt, weight: "bold")[catanrl v2]
  #v(-0.5em)
  #text(size: 13pt)[experiment log]
  #v(0.1em)
  #text(size: 9pt, fill: muted)[living document · last updated 26 october 2026 · #link("https://github.com/mcrco/catanrl-v2")[github] · #link(wandb-project)[wandb]]
]
#v(0.4em)

#outline(depth: 2)
#pagebreak()

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
