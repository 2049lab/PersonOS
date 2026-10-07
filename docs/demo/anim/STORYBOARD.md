# Who Ate the Halloween Candy? — PersonOS demo storyboard (v5)

Hand-drawn picture-book animation (p5.js + p5.brush, frame-stepped, deterministic).
Original characters and code. An intro video: show that PersonOS **recognises people and
corrects itself**, without exposing internal chain machinery.

## Rules

1. Subtitles = **dialogue only** (speaker + line, EN with ZH beneath). Audience view, real names.
   No narration — the acting and Pebble's notebook explain everything.
2. **Normal speed.** One line at a time, ≥2.5 s per line + ~0.6 s pause; every notebook change
   gets time to be read.
3. **Story logic airtight and fun**: every clue is planted before it pays off (Mum's rule → motive;
   ketchup splat → the stain; Leo leaving his sticky sheet on the sofa arm → the swap in the dark;
   sticky ketchup → wrappers stuck to the sheet → Mum's accusation; Biscuit asleep → alibi). Nobody does anything without a reason.
4. Notebook (always visible after the cover) has two parts:
   - **Who's who** — one card per person Pebble knows about. The **screenplay codes** (`P1`, `P2`…)
     that refer to that person sit on the card as small chips.
   - **Screenplay** — what happened, written immediately, always with codes (`P2: Mia, the candy's
     over there!`, `(P3 tosses a candy to P2)`), never with names while the story runs.
   - Only at the end of the night (**"tonight's memory saved"**) does an ink-wipe rewrite the
     screenplay through Who's who (`P1 → Mia`, …) — the result is one complete screenplay with the
     right names, which is what Pebble answers from in the morning.
5. Readability: one notebook change at a time (highlighter swipe + Pebble's lens glances at it);
   ≤6 visible screenplay lines (older ones scroll up and fade); each person has one colour used
   everywhere (stage tag, card, code chip, subtitle name); unknown = grey.

## Layout (1920×1080)

Stage ≈ 64% left, notebook ≈ 36% right, subtitle strip under the stage only.

```
┌────────────────────────────────┬──────────────────────────────┐
│                                │ PEBBLE'S MEMORY              │
│                                │ ─ Who's who ──────────────── │
│            STAGE               │ [face] Mia?   (P1)  👟 🔊     │
│                                │        called "Mia" by P2    │
│                                │ [face] Leo    (P2)  🧦 📏 🔊  │
│                                │ [face] ?      (P3)  🔔        │
│                                │ [face] Mum ✓ known           │
│                                │ ─ Screenplay · 21:02 ─────── │
│                                │ P2: Whoa… Mia, the candy's…  │
├────────────────────────────────┤ (P3 tosses a candy to P2)    │
│ Leo: Whoa… Mia, the candy's…   │                              │
│ Leo：哇……Mia，糖在那边！         │                              │
└────────────────────────────────┴──────────────────────────────┘
```

**Who's who cards**: avatar doodle (face; ghost doodle after costumes) · name field · code chips ·
cue icons (pink sneakers / striped socks / height / bell / ears / voice), ~1.5× previous size,
lit in the person's colour when they're the evidence, faint grey otherwise · one evidence line
(`called "Mia" by P2`, `self-introduction`, `same pink sneakers`).
Name field states, visually distinct:
- `?` unknown (grey card)
- `Mia?` guess — pencil grey + small confidence bar; chip attached with a dashed outline
- `Mia` sure — ink + small red seal placed right of the name (never over text)
A known person from before (Mum) has a card with `✓ known` from the start.

**Corrections** happen only on the cards, and are visible as motion:
- a code chip **flies** from one card to another (lift → arc → land with a little bounce);
- a card that turns out to be a duplicate folds into the right card;
- a conflict = the card flashes red with a small tag (`⚠ Leo in two places?`).
The screenplay text never changes until the end-of-night rewrite.

Stage: speaker's mouth animates + tiny tick near the head; head tags show the card's current
name (`P1`, `Mia?`, `Mia`). Off-screen voice = small voice-wave icon at the kitchen door.
No big speech bubbles. Subtitle fully legible for the whole line, never over a face.

## Cast

| Character | Look | Cues |
|---|---|---|
| Pebble (camera) | cream egg robot, one lens eye, witch hat, pumpkin sticker, antenna bulb | — |
| Mia | smallest, pigtails, orange pumpkin sweater, **pink sneakers**, high giggly voice | sneakers, voice |
| Leo | tallest, violet hoodie, floppy messy fringe (no spikes), **green striped socks** | height, socks, voice |
| Sam | round, mint cat-tee, **bell on his shoe**, giggles | bell |
| Biscuit | sausage dog, orange bandana | ears, tail, bark |
| Mum | curlers, apron — already known to Pebble | face, voice |

Ghost form = identical white sheet, eye holes; cue peeks under the hem.

## Script (target ~120 s)

### Cover (0–4 s)
Picture-book cover "Who Ate the Halloween Candy?", Pebble peeks over it, page turns.

### 21:02 — Arrival
| Stage | Subtitle | Screenplay | Who's who |
|---|---|---|---|
| Doorbell; Pebble rolls to the door. Biscuit asleep in his basket. | **Pebble:** Coming! / 来啦！ | `21:02` · `(doorbell rings)` · `Pebble: Coming!` | card `Mum ✓ known` already there |
| Door bursts open; three kids tumble in. | — | `(three kids burst in)` | new grey cards `? (P1)` pink sneakers · `? (P2)` tall, striped socks · `? (P3)` bell |
| Mia jumps. | **Mia:** Trick or treat! / 不给糖就捣蛋！ | `P1: Trick or treat!` | P1 voice lit |
| Leo points at the bowl, facing Mia. | **Leo:** Whoa… Mia, the candy's over there! / 哇……Mia，糖在那边！ | `P2: Whoa… Mia, the candy's over there!` | P1 → `Mia?` (`called "Mia" by P2`) |
| Mia dashes to the bowl. | **Mia:** Mine! / 我的！ | `P1: Mine!` · `(P1 runs to the candy bowl)` | — |
| Sam tosses a candy. | **Sam:** Leo, catch! / Leo，接着！ | `P3: Leo, catch!` · `(P3 tosses a candy to P2)` | P2 → `Leo?` |
| Leo reaches, trips on the rug. | **Leo:** Oof! / 哎哟！ | `P2: Oof!` · `(P2 trips over the rug)` | P2 → `Leo` sure (`answered to "Leo"`) |
| Sam giggles, bell jingles. | **Sam:** Hehe — I'm Sam, by the way. / 嘿嘿——对了，我叫 Sam。 | `P3: Hehe — I'm Sam, by the way.` | P3 → `Sam` sure (`self-introduction`) |
| Mia, mouth full, points at the basket. | **Mia:** Shh — Biscuit's sleeping. / 嘘——Biscuit 在睡觉。 | `P1: Shh — Biscuit's sleeping.` · `(a dog sleeps in the basket)` | + card `Biscuit (P4)` dog (`named by P1`); P1 → `Mia` sure (`called "Mia" again, same voice`) |

### 21:08 — Mum's rule, costumes, the ketchup
| Stage | Subtitle | Screenplay | Who's who |
|---|---|---|---|
| Mum leans in from the kitchen with hot dogs + ketchup, sets them on the side table. | **Mum:** One candy each tonight — the rest is for tomorrow! / 今晚每人只能吃一颗，剩下的留到明天！ | `21:08` · `P5: One candy each tonight — the rest is for tomorrow!` · `(P5 puts hot dogs and ketchup on the table)` | chip `P5` flies straight onto `Mum ✓ known` (face + voice) — recognised instantly, no new card |
| Kids groan; Mia pouts and eyes the bowl. Mum leaves. | **Mia, Leo, Sam:** Awww… / 啊——…… | `(P1 P2 P3 groan)` · `(P5 leaves)` | — |
| The hot-dog smell wakes Biscuit; he sniffs his way behind the sofa. | — | `(P4 leaves the basket)` | — |
| Mia pulls out sheets with a mischievous grin. | **Mia:** Costume time! / 换装时间到！ | `P1: Costume time!` | — |
| Sheets fly up — three ghosts. A spare little sheet flutters off the pile behind the sofa. | — | `(P1 P2 P3 put on white sheets)` | small chip `faces visible: 0`; avatars → ghost doodles; names stay; evidence → `same pink sneakers` / `tallest, striped socks` / `bell` |
| A tiny ghost trots out from behind the sofa. | — | `(P6, a small ghost, appears)` | new grey card `? (P6)` small ghost — `new person?` |
| Sam-ghost waves at it. | **Sam:** Boooo! / 呜——！ | `P3: Boooo!` | bell lit |
| Its ears pop out, tail wags. | **Biscuit:** Arf! / 汪！ | `P6: Arf!` | **correction:** the `? (P6)` card folds into `Biscuit` — chip `P6` lands on Biscuit's card (`ears, tail, bark`) |
| Leo-ghost stumbles blind into the side table — ketchup squirts a big red splat on his sheet. | **Leo:** Aw, ketchup! I can't see a thing in this. / 啊，番茄酱！套着这个啥也看不见。 | `P2: Aw, ketchup! I can't see a thing in this.` · `(ketchup splats on P2's sheet)` | Leo's card: `+ ketchup stain` |
| Leo pulls the sheet off, holding it at arm's length; face visible again. He drapes it over the sofa arm. | **Leo:** Ew, it's all sticky! I'll put it back on later. / 咦，黏糊糊的！我待会儿再穿。 | `P2: Ew, it's all sticky! I'll put it back on later.` · `(P2 hangs the stained sheet on the sofa arm)` | Leo's avatar back to his face; evidence `stained sheet on the sofa arm` |
| Mia-ghost glances from the stained sheet on the sofa arm to the candy bowl and back; her eye holes narrow into a sly look; a tiny lightbulb-idea sparkle. | — | — | — |
| Biscuit-ghost curls up in the basket again. | — | `(P6 falls asleep in the basket)` | — |

### 21:14 — Blackout
| Stage | Subtitle | Screenplay | Who's who |
|---|---|---|---|
| Thunder; lights out; only eye holes + lens glow. | **Sam:** Eek! It's dark! / 啊！好黑！ | `21:14` · `(lights go out)` · `P3: Eek! It's dark!` | — |
| Pebble's lens switches to dim night view (dark blue, grainy, only soft silhouettes — no colours, no shoes visible). A small ghost silhouette slips its own sheet off, drops it on the floor, tiptoes to the sofa arm and pulls on the stained sheet. | — | `(P8, a small silhouette, drops a plain sheet on the floor)` · `(P8 takes the stained sheet from the sofa arm)` | Mia's card goes faint: `lost track in the dark`; new grey card `? (P8)` small silhouette |
| A tall silhouette (Leo) gropes along the sofa arm — nothing there — then finds the plain sheet on the floor and pulls it on. | — (only a muffled "hmm?" lost under the thunder — no usable voice) | `(P7, a tall silhouette, finds a plain sheet on the floor and puts it on)` | Leo's card goes faint: `lost track in the dark`; new grey card `? (P7)` tall silhouette |
| Lights back. The clean ghost (really Leo) shuffles toward the kitchen; the stained ghost (really Mia) tiptoes to the bowl. | — | `(lights come back)` · `(P7, plain sheet, walks into the kitchen)` · `(P8, ketchup-stained sheet, goes to the candy bowl)` | chip `P8` lands on **Leo's** card, dashed: `Leo? — his ketchup stain` (plausible but wrong) · chip `P7` waits grey |
| Off-screen from the kitchen. | **Leo:** Do we have orange juice? / 有橙汁吗？ | `P7 (kitchen): Do we have orange juice?` | chip `P7` lands on Leo's card (voice) → Leo's card flashes red: `⚠ Leo in two places?` |
| Stained ghost freezes mid-reach… a high giggle; Pebble's lens zooms to its feet: pink sneakers. | **Mia:** Hehehe… / 嘿嘿嘿…… | `P8: Hehehe…` | **correction:** chip `P8` lifts off Leo's card and flies to **Mia's** card (`high voice + pink sneakers`); Mia's evidence `wearing Leo's sheet` |

### 21:16–21:31 — The heist, then the night's memory is saved
| Stage | Subtitle | Screenplay | Who's who |
|---|---|---|---|
| Stained ghost sneaks back three times; the clock ticks; candy level drops; Biscuit snores. | **Mia (whisper):** One for me… one more for me… / 一颗给我……再一颗给我…… | `21:16 (P8 takes a candy)` · `21:22 (P8 takes a candy)` · `21:29 (P8 takes a candy)` · `21:31 (the candy bowl is empty)` | — |
| Close-up: candy wrappers get stuck to the sticky ketchup patch on the sheet each time she grabs. Mia drops the wrapper-covered stained sheet beside the empty bowl and tiptoes off, pocket bulging. Night falls; Pebble's antenna flashes. Hold ~5 s. | — | stamp `tonight's memory saved`; an ink-wipe runs down the whole screenplay (scrolling up from the top), rewriting codes through the cards: `P1/P8 → Mia`, `P2/P7 → Leo`, `P3 → Sam`, `P4/P6 → Biscuit`, `P5 → Mum` | all chips glow as their codes are rewritten |

Hold on the named screenplay long enough to read e.g. `Mia: Trick or treat!`,
`(Mia, ketchup-stained sheet, goes to the candy bowl)`, `(Mia takes a candy)`.

### Next morning 08:30 — Who did it?
| Stage | Subtitle | Notebook |
|---|---|---|
| Sunbeams. Mum holds the empty bowl; the stained sheet lies crumpled beside it, candy wrappers stuck all over the ketchup patch. | **Mum:** Who ate ALL the candy?! / 是谁把糖全吃了？！ | header `08:30`; head tags show names immediately (everyone is known now) |
| Mum lifts the sheet; wrappers dangle from the sticky ketchup patch. She peels one off and glares at Leo. | **Mum:** Candy wrappers stuck all over YOUR ketchup sheet, Leo! / 你那张番茄酱床单上粘满了糖纸，Leo！ | — |
| Leo, wide-eyed, hands up. | **Leo:** It wasn't me! / 不是我！ | — |
| Sam shakes his head, bell jingling. | **Sam:** Not me either! / 我也没有！ | — |
| Mia points at the dog. | **Mia:** Must've been… Biscuit? / 肯定是……Biscuit 吧？ | line `(Biscuit falls asleep in the basket)` 21:09 glows |
| Biscuit's ears shoot up. | **Biscuit:** Woof?! / 汪？！ | — |
| Mum turns to the robot. | **Mum:** Pebble, you saw everything. Who was it? / Pebble，你都看见了。到底是谁？ | — |
| Pebble rolls forward; the notebook scrolls to the cited lines (`Mia takes the stained sheet from the sofa arm`, `Mia, ketchup-stained sheet, goes to the candy bowl`, `Mia takes a candy` ×3, `the candy bowl is empty`), highlighted; a polaroid of pink sneakers under the stained sheet is clipped beside them. | **Pebble:** It was Mia. In the blackout she put on Leo's ketchup sheet — but her giggle and pink sneakers gave her away. / 是 Mia。停电时她换上了 Leo 那张沾番茄酱的床单——但她的笑声和粉色球鞋出卖了她。 | cited lines highlighted |
| A candy wrapper slips out of Mia's sweater pocket and flutters down in slow motion; everyone turns to her one by one. | **Mia:** …Hehe. Happy Halloween? / ……嘿嘿，万圣节快乐？ | — |
| Biscuit wags. | **Biscuit:** Woof! / 汪！ | — |

### End card (~5 s)
PersonOS lockup + tagline, `pip install personos`, github.com/2049lab/personos, Pebble winks.

## What each beat shows (for us, not on screen)

| Beat | Capability |
|---|---|
| Names learned from dialogue, guess → sure | naming from conversation, commit only with enough evidence |
| Mum recognised instantly | cross-session recognition of a known person |
| Ghosts keep their names | recognition beyond faces (shoes, height, bell, voice) |
| Tiny ghost folded into Biscuit | duplicate merged, pets too |
| Stained ghost: Leo? → Mia | contradiction detected and repaired |
| Screenplay in codes, rewritten at the end | facts recorded immediately, identities resolved at the end |
| Morning answer with cited lines | recall with evidence |
