// ==========================================
// 1. TEMPLATE DEFINITION
// ==========================================
#let academic-paper(
  title: "",
  abstract: none,
  authors: (),
  body,
) = {
  // Set PDF metadata
  set document(title: title, author: authors.map(a => a.name))

  // Configure page size, margins, and page numbering
  set page(
    paper: "a4",
    margin: (x: 1.5cm, y: 2.5cm),
    numbering: "1",
  )

  // Configure text formatting (standard academic font)
  set text(font: "New Computer Modern", size: 10pt)

  // Configure paragraph properties (justified text, standard indents)
  set par(justify: true, leading: 0.55em, first-line-indent: 0em)

  // Configure heading styles
  set heading(numbering: "I.A.1.")
  show heading: it => {
    set text(weight: "bold", size: 11pt)
    block(above: 1.5em, below: 1em)[#it]
  }

  // --- Render Title ---
  align(center, text(18pt, weight: "bold", title))
  v(1.5em)

  // --- Render Authors ---
  align(center, grid(
    columns: authors.len(),
    gutter: 2em,
    ..authors.map(a => [
      *#a.name*\
      #a.affiliation\
      //#link("mailto:" + a.email)
    ])
  ))
  v(2.5em)

  // --- Render Abstract ---
  if abstract != none {
    align(center)[
      #block(width: 85%)[
        *Abstract* \
        #v(0.5em)
        #align(left, text(style: "oblique", abstract))
      ]
    ]
    v(2em)
  }

  // --- Switch to Two-Column Layout ---
  show: columns.with(2, gutter: 1.5em)

  // --- Render Body Content ---
  body
}

// ==========================================
// 2. DOCUMENT USAGE
// ==========================================

#show: academic-paper.with(
  title: "A Standard A4 Two-Column Typst Template",
  abstract: [
    This document serves as a straightforward template for academic writing in Typst. It enforces an A4 paper size, a two-column grid, and sets up title, author, and abstract formatting conventions typical of scientific publishing.
  ],
  authors: (
    (
      name: "Researcher One",
      affiliation: "University of Applied Sciences",
      email: "author1@example.com",
    ),
    (
      name: "Researcher Two",
      affiliation: "Institute of Technology",
      email: "author2@example.com",
    ),
  ),
)

= Introduction
Typst provides a highly efficient environment for typesetting academic documents. This template establishes a standard double-column format utilizing the A4 paper size, which is commonly required for international conferences and journals.

== Motivation
Setting up structural rules early in a document ensures consistent output. By abstracting the layout logic into a function, the primary file remains clean and focused solely on the content.

= Methodology
We define the physical dimensions using the `#set page(paper: "a4")` rule. To organize the text into a standard dual-pane format, we apply the `show: columns.with(2)` rule immediately after the front matter.

#lorem(120)

== Secondary Considerations
The font is set to `New Computer Modern` to replicate standard scientific output. Justification and first-line indentation are also enforced.

#lorem(150)

= Conclusion
This configuration acts as a reliable starting framework for academic publishing in Typst, ensuring strict adherence to standard dimensional and layout requirements.
