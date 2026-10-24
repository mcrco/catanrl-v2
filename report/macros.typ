// shared colors and macros for report.typ and every file in sections/.
// each section starts with: #import "../macros.typ": *

#let accent = rgb("#1f4e8c")
#let muted = rgb("#5f5e5a")
#let rule = rgb("#bdbcb6")
#let note-fill = rgb("#f3f2ee")
#let code-fill = rgb("#f6f8fa")

#let btable(columns: auto, align: auto, header: (), ..rows) = table(
  columns: columns, align: align,
  table.header(table.hline(stroke: 0.8pt), ..header.map(h => text(weight: "bold", h)), table.hline(stroke: 0.5pt)),
  ..rows,
  table.hline(stroke: 0.8pt),
)
#let ci(lo, hi) = text(size: 0.88em, fill: muted)[#lo–#hi]
#let aside(body) = text(size: 0.88em, fill: muted, body)
#let mid = table.hline(stroke: 0.4pt + rule)
#let callout(body) = block(width: 100%, fill: note-fill, inset: (x: 9pt, y: 7pt), radius: 2pt, body)
#let todo = sym.ballot
#let done = sym.ballot.check
#let wandb-project = "https://wandb.ai/myang2-california-institute-of-technology-caltech/catanrl-v2"
#let wb(id, body) = link(wandb-project + "/runs/" + id, body)

// status chip, written inline after an experiment title: === title #chip("done")
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
  h(0.35em)
  box(
    stroke: 0.6pt + c,
    radius: 2pt,
    inset: (x: 3pt, y: 1.6pt),
    baseline: 1.6pt,
    text(size: 7.5pt, weight: "bold", fill: c, tracking: 0.04em, upper(s)),
  )
}

// one or two compact lines under an experiment title.
// dates-label names the dates field: "ran" by default, e.g. "queued" or "proposed" for work not run yet
#let meta(dates: none, dates-label: "ran", owner: none, config: none, wandb: (), records: ()) = {
  let field(name, value) = [#text(weight: "bold")[#name] #value]
  let as-list(v) = if type(v) == str { v.split(",").map(s => s.trim()).filter(s => s != "") } else { v }
  let fields = ()
  if dates != none { fields.push(field(dates-label, dates)) }
  if owner != none { fields.push(field("idea", owner)) }
  if config != none { fields.push(field("config", as-list(config).map(raw).join(", "))) }
  if wandb.len() > 0 {
    let links = wandb.map(w => if type(w) == str { wb(w, w) } else { wb(w.at(0), w.at(1)) })
    fields.push(field("wandb", links.join(", ")))
  }
  if records.len() > 0 { fields.push(field("records", records.map(raw).join(", "))) }
  if fields.len() > 0 {
    block(above: 0.45em, below: 0.75em, sticky: true, {
      set text(size: 8.5pt, fill: muted)
      set par(justify: false, leading: 0.5em)
      fields.join([ · ])
    })
  }
}

// plain-language summary at the top of an experiment
#let tldr(body) = callout[*tl;dr.* #body]

// the command(s) that reproduce an experiment
#let repro(body) = block(
  width: 100%, fill: code-fill, radius: 3pt, inset: (x: 8pt, y: 6pt), above: 0.9em, below: 1.1em,
  {
    set text(size: 8.5pt)
    set par(justify: false)
    text(size: 7.5pt, weight: "bold", fill: muted, tracking: 0.04em)[reproduce]
    v(0.35em, weak: true)
    body
  },
)
