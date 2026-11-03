// Shared colors, fonts and macros for report.typ and every file in sections/.
// Each section starts with: #import "../macros.typ": *

#let accent = rgb("#1f4e8c")
#let muted = rgb("#5a5955")
#let rule = rgb("#bdbcb6")
#let hair = rgb("#e4e3df")
#let note-fill = rgb("#f3f2ee")
#let group-fill = rgb("#eef1f5")
#let code-fill = rgb("#f6f8fa")

// Body text is serif; tables, captions, metadata and headings are sans with tabular figures.
#let serif = "Libertinus Serif"
#let sans = "Source Sans 3"
#let mono = "DejaVu Sans Mono"

#let wandb-project = "https://wandb.ai/myang2-california-institute-of-technology-caltech/catanrl-v2"
#let wb(id, body) = link(wandb-project + "/runs/" + id, body)

// Booktabs-style table: rules at the top, under the header and at the bottom, hairlines between rows.
#let btable(columns: auto, align: auto, header: (), ..rows) = table(
  columns: columns, align: align,
  stroke: (x, y) => if y > 1 { (top: 0.4pt + hair) },
  table.header(table.hline(stroke: 0.9pt), ..header.map(h => text(weight: "semibold", h)), table.hline(stroke: 0.6pt)),
  ..rows,
  table.hline(stroke: 0.9pt),
)
// A shaded row that names a group of rows: #tgroup(5)[2 players, BASE, 15 VP]
#let tgroup(cols, body) = table.cell(colspan: cols, fill: group-fill, text(weight: "semibold", body))
// A 95% interval, printed after a number: 94.96 #ci[94.3][95.5]
#let ci(lo, hi) = { h(0.25em); text(size: 0.84em, fill: muted)[(#lo–#hi)] }
// A second, quieter line in a table cell, e.g. the checkpoint under an agent's name: [Raw policy \ #sub[`run@e100`]]
#let sub(body) = text(size: 0.88em, fill: muted, body)
#let aside(body) = text(size: 0.9em, fill: muted, body)
#let mid = table.hline(stroke: 0.6pt + rule)
#let callout(body) = block(width: 100%, fill: note-fill, inset: (x: 10pt, y: 8pt), radius: 3pt, body)
#let todo = sym.ballot
#let done = sym.ballot.check

// Status chip, written after an experiment title: === Title #chip("done")
#let chip-colors = (
  done: rgb("#2e7d32"),
  running: accent,
  queued: accent,
  stopped: rgb("#9a6700"),
  todo: muted,
  parked: muted,
)
#let chip(s) = {
  assert(s in chip-colors, message: "chip: unknown status " + repr(s))
  let c = chip-colors.at(s)
  h(0.4em)
  box(
    stroke: 0.6pt + c,
    radius: 2pt,
    inset: (x: 3.5pt, y: 1.8pt),
    baseline: 1.8pt,
    text(font: sans, size: 7.5pt, weight: "bold", fill: c, tracking: 0.05em, upper(s)),
  )
}

#let capitalize(s) = if s.len() == 0 { s } else { upper(s.first()) + s.slice(s.first().len()) }
#let owner-names = (marco: "Marco", assistant: "Claude")

// One or two compact lines under an experiment title.
// dates-label names the dates field: "ran" by default; e.g. "queued", "built" or "proposed" for work not run yet.
#let meta(dates: none, dates-label: "ran", owner: none, config: none, wandb: (), records: ()) = {
  let field(name, value) = [#text(weight: "semibold", fill: black.lighten(15%))[#name:] #value]
  let as-list(v) = if type(v) == str { v.split(",").map(s => s.trim()).filter(s => s != "") } else { v }
  let fields = ()
  if dates != none { fields.push(field(capitalize(dates-label), dates)) }
  if owner != none { fields.push(field("Idea", owner-names.at(owner, default: owner))) }
  if config != none { fields.push(field("Config", as-list(config).map(raw).join(", "))) }
  if wandb.len() > 0 {
    let links = wandb.map(w => if type(w) == str { wb(w, w) } else { wb(w.at(0), w.at(1)) })
    fields.push(field("W&B", links.join(", ")))
  }
  if records.len() > 0 { fields.push(field("Records", records.map(raw).join(", "))) }
  if fields.len() > 0 {
    block(above: 0.5em, below: 0.8em, sticky: true, {
      set text(font: sans, size: 8.8pt, fill: muted)
      set par(justify: false, leading: 0.5em)
      show raw: set text(size: 1.02em)
      fields.join([#h(0.5em)·#h(0.5em)])
    })
  }
}

// The plain-language summary at the top of an experiment.
#let tldr(body) = callout[*Summary.* #body]

// The command(s) that reproduce an experiment.
#let repro(body) = block(
  width: 100%, fill: code-fill, radius: 3pt, inset: (x: 9pt, y: 7pt), above: 1em, below: 1.2em,
  {
    set text(size: 8.6pt)
    set par(justify: false)
    text(font: sans, size: 7.8pt, weight: "bold", fill: muted, tracking: 0.05em)[REPRODUCE]
    v(0.4em, weak: true)
    body
  },
)
