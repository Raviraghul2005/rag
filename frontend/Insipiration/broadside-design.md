# Broadside — Design System

*A gritty, editorial poster system extracted from the "Graphic Design Prompts for Claude" carousel.*

The whole personality comes from three moves: **warm paper stock** with a light grain, **ultra-condensed black caps** that shout, and **one burnt-rust accent** rationed to keywords and a hand-painted sunburst. It's built for swipe-through content — every slide is a single-sheet poster.

---

## 1. Principles

1. **Type is the image.** The condensed caps are big enough to be the artwork. Photography sits behind them, always desaturated.
2. **One accent, kept rare.** Rust means "read this." Highlight the two or three load-bearing words in a headline and nothing else.
3. **Paper, not white.** The background is a warm stock with grain. Pure white reads cheap next to it.
4. **Grit is texture, not decoration.** Grain, sunburst, and edge motifs (barbed wire in the source) add roughness — they never carry meaning.

---

## 2. Color

Warm paper, black ink, one burnt-rust accent.

### Paper stock
| Token | Hex | Use |
|---|---|---|
| `--paper-100` | `#f7f4ec` | Cards, phone screens, lightest |
| `--paper-200` | `#f1ede3` | Page background base |
| `--paper-300` | `#e7e1d4` | Recessed panel |
| `--paper-400` | `#dcd5c5` | Borders on paper, muted fills |

### Ink & neutrals
| Token | Hex | Use |
|---|---|---|
| `--ink` | `#16120e` | Display caps, body — warm near-black |
| `--ink-soft` | `#2c2721` | Secondary headings |
| `--muted` | `#8c867a` | Meta, page counter, handle |
| `--ghost` | `#d8d2c5` | Giant ghosted slide number |

### Rust — the one accent
| Token | Hex | Use |
|---|---|---|
| `--rust` | `#bb5230` | Highlighted keywords, sunburst, key marks |
| `--rust-deep` | `#97401f` | Pressed / hover |
| `--rust-tint` | `#e2bca6` | Soft tag fills |

> The accent is a **burnt sienna** — redder and earthier than a soft clay orange. Keep it there; a lighter clay drifts toward a generic "AI cream-and-terracotta" look.

---

## 3. Typography

Five voices, strict hierarchy.

| Role | Face | Weight | Setting | Where |
|---|---|---|---|---|
| **Poster** | Anton | 400 | Uppercase, line-height `.86`, tracking `.01em` | 1–3 word titles, cover words, section heads |
| **Statement** | Archivo | 800 | Tight, keywords in rust | Multi-word headlines |
| **Serif** | Lora | 500 italic | One line | The human aside under a poster title |
| **Script** | Kaushan Script | 400 | Slight rotation | A single flourish per slide |
| **Body** | Inter | 400 (italic for quotes) | 16px / `1.55` | Paragraphs, UI, prompt text |
| **Label** | Archivo | 800 | Uppercase, tracking `.04–.16em` | "WHY IT WORKS:", eyebrows |

**Rules of thumb**
- The poster face never runs more than a few words. For a sentence, switch to the Archivo statement style.
- In a statement headline, colour only the load-bearing words rust; the rest stays ink.
- The serif and script are accents — one line each, never a paragraph.

### Type scale
```
--step-caption   : 0.75rem
--step-small     : 0.875rem
--step-body      : 1rem
--step-lead      : 1.1875rem
--step-h2        : 1.5rem
--step-h1        : 2.25rem
--step-statement : clamp(1.75rem, 4vw, 2.75rem)
--step-poster    : clamp(3rem, 11vw, 7rem)
```

---

## 4. Space & radius

```
--s-1  4px    --s-4 16px    --s-7 48px
--s-2  8px    --s-5 24px    --s-8 64px
--s-3 12px    --s-6 32px    --s-9 96px

--r-sm   6px   (tags)      --r-lg   20px  (panels, quote box)
--r-md  12px   (cards)     --r-pill 999px (buttons)
```

Margins stay generous so the paper breathes; corners stay softly rounded.

---

## 5. Signature devices

Four devices give the system its grit. Use them one at a time.

- **Rust sunburst** — hand-painted rays radiating from behind the subject. In CSS, a `repeating-conic-gradient` in `--rust`, masked with a radial fade and rotated a few degrees off-axis so it doesn't feel mechanical.
- **Ghosted number** — the slide's index set huge in `--ghost`, bottom-right, behind the text. This is the numbering device — only use it where the order carries meaning (a ranked list of prompts).
- **Rust keyword highlight** — inside an ink statement, the two or three key words flip to `--rust`.
- **Paper grain** — an SVG `feTurbulence` tile at ~5% opacity, `mix-blend-mode: multiply`, over the whole surface.
- **Edge grit (asset)** — the source frames slides with barbed wire. Treat this as an optional photographic edge overlay, desaturated, not a CSS effect.

### Sunburst recipe
```css
.sunburst{
  position:absolute; inset:-30%;
  background:repeating-conic-gradient(from 6deg at 50% 50%,
    var(--rust) 0 6deg, transparent 6deg 19deg);
  -webkit-mask-image:radial-gradient(circle at 50% 50%, #000 22%, transparent 62%);
  mask-image:radial-gradient(circle at 50% 50%, #000 22%, transparent 62%);
  transform:rotate(-4deg);
  opacity:.85;
}
```

---

## 6. Components

Built entirely from the tokens above.

- **Buttons** — pill-shaped, 2px border. *Ink* is default, *rust* is the loud CTA (e.g. "Comment"), *ghost* is quiet. The "Save post" button pairs a bookmark glyph.
- **Tags** — rust-tint for topic labels, solid ink for the page counter.
- **Prompt box** — a 2px ink border, `--r-lg` corners, italic body. Where a copy-paste prompt lives.
- **Label + body** — a heavy tracked-caps label ("WHY IT WORKS:") over an Inter paragraph.
- **Page counter** — `n / 11`, top-right on every slide.
- **Handle** — muted sign-off, bottom-right.

### Elevation
```
--shadow-sm : 0 4px 14px rgba(22,18,14,.10)
--shadow-md : 0 18px 44px rgba(22,18,14,.16)
```

---

## 7. Slide templates

A whole carousel is built from four archetypes. Every post is a **cover → content cards → CTA**; the statement slide is optional.

| Template | Structure |
|---|---|
| **Cover** | Kicker + counter · poster title (one word may go rust) · serif subtitle · script flourish · subject on sunburst · handle |
| **Statement** | Heavy Archivo headline with 2–3 rust keywords · content grid (phone mockups / cards) · "Save post" CTA |
| **Content card** | Poster title · bordered prompt box · "WHY IT WORKS:" label + body · giant ghosted rank number |
| **CTA** | Stacked poster caps (one word rust) · a supporting heavy line ("for the full copy-paste version") |

Instagram portrait ratio is **4:5** (1080 × 1350). Keep the counter and handle in the same corners on every slide so the reader stays oriented.

---

## 8. Accessibility floor

- Ink on paper clears WCAG AA comfortably; use `--muted` only for meta and large text.
- Rust on paper is strong enough for headline keywords; don't set long body copy in rust.
- Focus is always visible: 2px `--rust` ring, 3px offset.
- Layout collapses to one column on mobile; `prefers-reduced-motion` disables transitions.

---

## 9. Do / don't

**Do**
- Let one condensed word be the biggest thing on the slide.
- Highlight only the words that carry the point.
- Keep every photo desaturated.

**Don't**
- Spread rust across a whole headline or into body copy.
- Run the poster face for full sentences — switch to the statement style.
- Use pure white anywhere; stay on paper.
- Let the serif or script run longer than a single line.

---

*Source: "Graphic Design Prompts for Claude" carousel by @sifuyik. This system generalizes its palette, type and layout for reuse; the specific prompts, copy and artwork remain the author's.*
