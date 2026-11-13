# Epic 2 · AI x Epistemics Research Sprint — Brief and Rules

From Apart's joining email (received 2026-11-12). Online, Friday 13 to Sunday 15 November 2026. Our project: [Phase 2 plan](epic2_phase2_plan_12-11-2026.md).

## Key dates (UTC)

| When | What |
| --- | --- |
| Fri 13 Nov, 14:15 | Sprint starts with the logistics talk |
| Mon 16 Nov, 11:59 | **Submissions close** (Sunday 15 Nov, 11:59 PM Anywhere on Earth) |

## Rules that affect us

- **AI tools** are allowed in any way. The team is responsible for everything in the report; made-up results or references can disqualify a project.
- **Existing work** is encouraged, but the report must say clearly what is new during the sprint, and code written before the sprint must be marked as preexisting **in the README and in the report**. Undisclosed prior work can lead to disqualification. For us: everything in `experiments/epic2/phase1_confidence/` (Phases 1 and 1B, 3 to 12 November) is preexisting; Phase 2 code goes in a new folder.
- **Report:** a PDF on the official template (Guidelines tab), **4 to 8 pages** excluding references and appendices; 8 pages is a hard cap. A **Limitations and Dual-Use / Ethical Considerations appendix is required**.
- **Judging:** scored on the rubric below. A judge can decline to score a report too unclear to follow.
- **Submitting:** projects are not shown on the website until after judging, so check every detail before submitting. To replace a submission, submit again with **exactly the same title**.
- **Teams:** 1 to 5 people works best; solo is fine. Each person can be on at most two projects. Teams form in the #projects | teams channel on Discord.
- **Prizes:** $2,000 in cash across all tracks; top teams are fast-tracked to the Apart Fellowship. The sponsor Adaption offers each participant 1,250 credits and three months of its Builder plan (sign-up link in the email).

## Judging rubric (Guidelines tab, 2026-11-13)

All projects are scored on the same rubric whatever the track. Three dimensions, 1 to 5 each:

| Dimension | 3 | 4 | 5 |
| --- | --- | --- | --- |
| **Impact Potential & Innovation**: how much it would matter if it worked; how innovative | Clear problem, reasonable approach, some novelty in framing or method | Important problem with an original approach, or a neglected problem area; others could build on it | Critical problem, genuinely novel approach or a new direction; clear theory of change |
| **Execution Quality**: soundness of method, implementation, findings | Technically solid for the time; interpretable results; limitations acknowledged | Thorough method, convincing validation, results clearly support conclusions; immediately useful | Ambitious scope executed rigorously; surprising findings, novel methods or unusually robust validation |
| **Presentation & Clarity** | Problem, method, findings, limitations all present without undue effort | Easy to follow, well structured, right level of detail | A pleasure to read; complex ideas made accessible |

Low end: Impact 2 = "a real problem but with a generic or well-trodden approach"; Presentation 2 = "key information buried... or diluted by excessive length". **For 4 or 5 on Impact, judges ask: "is this actually new to the field, or replicating recent work?"**

**What this means for our report.**

- *Impact* is the dimension at risk. "Models know more than they say" is well trodden, and Sanz-Guerrero et al. already have ownership bias. The report has to lead with what is new (self-knowledge depends on how the answer is produced; correct vs wrong by construction; the probe on the same answers; whether the user-framing fix reaches computation and perception) and state the theory of change: which confidence readout deployers and evaluators can trust, and why stated confidence fails.
- *Execution* is our strength: validated grading, difficulty controls, paired CIs, a probe ceiling, pre-registered predictions, a `verify.py`. The rubric asks for variance and baselines in the results.
- *Presentation*: one claim, one key figure on page 1, short. Excess length is penalised explicitly.

## Submission requirements

- **Required:** PDF on the official template (Guidelines tab, not the one in any email); at most 8 pages excluding references and appendices; title and an **abstract of 150 to 250 words**; author names and affiliations; the **Limitations and Dual-Use / Ethical Considerations** appendix: what the result does and does not establish, any way the tool or findings could be used to manipulate rather than inform, and how personal data was handled (none here).
- **Optional:** a public GitHub repo; a 3 to 5 minute video demo.
- **Recommended structure:** Introduction (question, why it matters); Related Work; Methodology (models, prompts, datasets, rubrics, metrics, enough to replicate); Results (quantitative, with variance and baselines); Discussion (implications, limitations, future work); the required appendix; References. Any consistent citation style.
- **Pre-submission checklist:** template PDF, at most 8 pages before references and appendices, abstract 150 to 250 words, authors and affiliations, the appendix, links work. After submitting, check the confirmation email has the right PDF. Unfinished work is better than none; judges score what was done, and honest limitations are welcome.
- **Judging:** expert judges, about a week after the event; results 1 to 2 weeks after that, with reviewer feedback by email. Support: Discord #help-desk or sprints@apartresearch.com.

## Talks (UTC, on Zoom)

RSVP to each on Luma to get the Zoom link and a reminder; nobody is added automatically. All talks are recorded and posted on Discord and the Schedule tab.

| Day | Time | Speaker |
| --- | --- | --- |
| Fri 13 Nov | 14:15 | Kamil Alaa, Apart Research: logistics |
| Fri 13 Nov | 15:15 | Keivan Navaie, Lancaster University and MATS |
| Fri 13 Nov | 17:15 | Workshop with Adaption: Bargav Jagatha |
| Fri 13 Nov | 18:15 | Dean Issacharoff, VSIN and Mozilla Democracy x AI Incubator |
| Fri 13 Nov | 20:45 | Keynote: Owen Cotton-Barratt, Forethought and Differential Labs |
| Sat 14 Nov | 14:15 | Jonathan Stray, CHAI, UC Berkeley |
| Sat 14 Nov | 18:15 | Joel Suro, OMNA |
| Sun 15 Nov | 15:15 | Orestis Palermos, University of Ioannina |

The user plans to attend most talks, so the work plan keeps those slots free of tasks that need them (4090 runs can go in the background).

## Where things are

- **Sprint page tabs:** Overview (tracks, prizes), Resources (reading, example projects and datasets by track; check before looking elsewhere), Guidelines (rules, rubric, template, FAQ), Schedule (talks and recordings).
- **Discord:** team formation, announcements, and #help-desk (tag Support).

## To do before writing

- [x] Rubric recorded above (2026-11-13).
- [ ] Download the template from the Guidelines tab.
- [ ] Check the Resources tab's reading list for our track against the Phase 2 study list.
- [ ] Confirm the track name for our project.
