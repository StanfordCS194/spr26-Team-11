// =============================================================================
// Atlas — Mock Query Data
// =============================================================================
//
// This file is the prototype's source of truth. There is no backend and no
// API calls anywhere in Atlas — the entire app is driven from the three
// pre-baked MockQuery objects defined below. A "query" here models the full
// end-to-end response you'd expect Atlas to return if it really did hit Mail,
// Messages, Documents, and Calendar on your machine:
//
//   - A synthesized prose `answer` (used by the AtlasAnswer block, Subtask 5).
//   - A short ranked list of `results` (the Top K list, Subtask 6).
//   - Per-result body text, metadata, and highlights (the expanded preview
//     below the list, Subtask 7).
//
// Shape decisions:
//   - The 3 queries deliberately exercise different source-type mixes so that
//     the UI gets stress-tested across every rendering path:
//       Query 1: email-heavy      → 2 Mail + 1 Calendar + 1 Documents
//       Query 2: calendar-heavy   → 2 Calendar + 1 Mail + 1 Documents
//       Query 3: mixed Docs+iMsg  → 2 Documents + 1 Messages + 1 Mail
//     Every query touches at least 3 of the 4 source categories, per spec.
//
//   - Matching is a case-insensitive SUBSTRING test on the `trigger`. Typing
//     "bud" reveals the "q2 budget" query; typing "onboard" reveals the
//     onboarding query. See findMatchingQuery() at the bottom.
//
//   - All dates are absolute strings (no "yesterday", no "next Thursday") so
//     the prototype reads the same way today or six months from now. Content
//     is dated early-to-mid April 2026 to feel current as of the demo.
//
// =============================================================================

// -----------------------------------------------------------------------------
// Type definitions
// -----------------------------------------------------------------------------
// Source category a result belongs to. Drives the chip colour/label in the
// results list AND the "Open in [app]" link in the expanded preview. Kept as
// a string literal union (rather than enum) so it round-trips cleanly through
// JSON if we ever serialise to the log file or a mock API boundary.
export type SourceType = "Mail" | "Messages" | "Documents" | "Calendar";

// One row in the Top K list AND the data backing the expanded preview below
// it. We use a single shape for both views because the prototype is small
// enough that splitting "list item vs detail" would just add indirection —
// the fields not used by the list (body, highlights, openInApp) are free
// within the preview and ignored elsewhere.
export interface MockResult {
  // Stable identity for selection state and logging. Format: "q<queryId>-r<n>".
  id: string;

  // Which surface this fake result came from. Determines the chip + icon
  // styling and the "Open in __" target app.
  source: SourceType;

  // First line shown in the list row AND as the preview header. Should read
  // like a real subject line / event title / file name / message preview.
  title: string;

  // Secondary line in the list row: snippet / attendee list / file path /
  // preceding-message-author — whatever makes sense for the source type.
  subtitle: string;

  // "From" field surfaced in the preview metadata table. For Calendar this
  // is the organiser; for Messages, the other party; for Documents, the
  // author/owner.
  from: string;

  // Human-readable absolute date. Format chosen to match Apple's Mail/
  // Calendar style: "MMM d, yyyy · h:mm a". Kept as a string (not Date)
  // because this value is only ever displayed, never compared.
  date: string;

  // Full preview body shown under the header of the expanded preview. For
  // Mail this is the email body; for Calendar, the event description/agenda;
  // for Documents, the first paragraph or summary; for Messages, a thread
  // snippet. Plain text — the renderer bolds `highlights` at display time.
  body: string;

  // Terms in `body` that should render bolded/highlighted when the preview
  // is shown. We store the literal terms (rather than offsets) because the
  // renderer does a per-term substring pass, which is robust to edits.
  highlights: string[];

  // Label for the "Open in ___ →" link at the bottom of the preview. Example
  // values per source type:
  //   Mail       → "Mail"
  //   Messages   → "Messages"
  //   Calendar   → "Calendar"
  //   Documents  → "Numbers" | "Pages" | "Figma" | "Code" | "Finder" (varies)
  // Since the prototype doesn't actually open anything, this is purely a
  // label; but picking native-macOS app names keeps the illusion believable.
  openInApp: string;
}

// One pre-baked query. Typing any case-insensitive substring of `trigger`
// surfaces this query's answer + results.
export interface MockQuery {
  // Stable identity for logging. Format: "q<n>".
  id: string;

  // The "canonical" phrase a user might type. Matching is substring-based
  // and case-insensitive, so short inputs like "bud" or "onboard" will
  // match "q2 budget" and "onboarding" respectively.
  trigger: string;

  // Atlas's synthesized prose answer for the AtlasAnswer block. Aim: 2–4
  // sentences, conversational, name-checks people and concrete details.
  // Should read like a senior teammate summarizing the situation, not like
  // a search engine echo.
  answer: string;

  // Ranked Top K — 4 entries, spread across ≥3 source types. Order matters:
  // the first entry is the default selection when the query resolves.
  results: MockResult[];
}

// =============================================================================
// The three pre-baked queries.
// =============================================================================
//
// Casting notes:
//   Q1 characters — SaaS/B2B finance & engineering world.
//   Q2 characters — product design team.
//   Q3 characters — engineering onboarding.
// Names chosen to be phonetically varied and unambiguous in speech (no two
// names share a first initial within a single query), which makes demo
// screen-sharing less confusing.
// =============================================================================

export const mockQueries: MockQuery[] = [
  // ---------------------------------------------------------------------------
  // Query 1 — EMAIL-HEAVY.
  // Scenario: execs finalizing a Q2 budget. 2 Mail threads dominate, with a
  // calendar placeholder for the upcoming review and the live spreadsheet
  // version as a document result.
  // ---------------------------------------------------------------------------
  {
    id: "q1",
    trigger: "q2 budget",
    answer:
      "Sarah proposed a 12% engineering headcount increase for Q2, weighted toward " +
      "backend and data platform. Miguel flagged that data platform's allocation " +
      "may be undersized given the ML roadmap; Priya suggested deferring the " +
      "customer success hire to Q3 to free up about $180K. Final numbers are being " +
      "reviewed ahead of Thursday's exec meeting.",
    results: [
      {
        id: "q1-r1",
        source: "Mail",
        title: "Re: Q2 budget proposal — headcount revisions",
        subtitle:
          "Sarah Chen — I've updated the headcount section based on Miguel's feedback…",
        from: "Sarah Chen <sarah.chen@northwind.io>",
        date: "Apr 15, 2026 · 11:23 AM",
        body:
          "Team — I've updated the Q2 budget headcount section based on Miguel's " +
          "feedback from Monday. Backend gets +3 (two senior, one mid), data " +
          "platform gets +2 (both senior). That's the 12% growth we discussed. " +
          "I've left the customer success hire in for now but tagged it as " +
          "deferrable if we need to absorb the data platform ask Miguel raised. " +
          "Numbers are in the v4 spreadsheet; happy to walk through it live on " +
          "Thursday before the exec review.",
        highlights: ["Q2 budget", "headcount", "data platform", "backend"],
        openInApp: "Mail",
      },
      {
        id: "q1-r2",
        source: "Mail",
        title: "Q2 budget — data platform concerns",
        subtitle:
          "Miguel Alvarez — the current allocation doesn't give us enough runway for the ML…",
        from: "Miguel Alvarez <miguel@northwind.io>",
        date: "Apr 16, 2026 · 2:47 PM",
        body:
          "Sarah — appreciate the revision, but I think the data platform " +
          "allocation is still undersized given the ML roadmap commitments " +
          "we made in the Q1 review. Two senior hires gets us to parity with " +
          "where we should have been last quarter, but it doesn't give us any " +
          "runway for the inference work. Could we discuss bumping it to +3 " +
          "and either deferring the CS hire (as Priya suggested) or pulling " +
          "from the contingency line? Happy to do a short sync before Thursday.",
        highlights: ["data platform", "ML roadmap", "Q2", "inference"],
        openInApp: "Mail",
      },
      {
        id: "q1-r3",
        source: "Calendar",
        title: "Q2 Budget Review — Exec",
        subtitle: "Thu Apr 25 · 3:00–4:30 PM · Boardroom / Zoom",
        from: "Sarah Chen (organizer)",
        date: "Apr 25, 2026 · 3:00 PM",
        body:
          "Agenda: (1) walk through the v4 Q2 budget spreadsheet; (2) resolve " +
          "the data platform headcount question Miguel raised; (3) decision on " +
          "the customer success hire timing; (4) sign-off for submission to " +
          "the board. Attendees: Sarah Chen, Miguel Alvarez, Priya Desai, and " +
          "the CEO. Pre-read: Q2_Budget_v4.xlsx (shared earlier this week).",
        highlights: ["Q2 budget", "data platform", "headcount"],
        openInApp: "Calendar",
      },
      {
        id: "q1-r4",
        source: "Documents",
        title: "Q2_Budget_v4.xlsx",
        subtitle: "Sarah Chen · ~/Documents/Finance/2026 · modified Apr 18",
        from: "Sarah Chen",
        date: "Apr 18, 2026 · 9:41 AM",
        body:
          "Q2 budget working spreadsheet. Tabs: Summary, Headcount, OpEx by " +
          "team, CapEx, Contingency. v4 incorporates Miguel's revised data " +
          "platform request (+2 senior engineers) and flags the customer " +
          "success hire as deferrable to Q3. Totals reconcile to the board " +
          "target within 1.2%. Locked for editing until Thursday's review.",
        highlights: ["Q2 budget", "headcount", "data platform"],
        openInApp: "Numbers",
      },
    ],
  },

  // ---------------------------------------------------------------------------
  // Query 2 — CALENDAR / MEETING-HEAVY.
  // Scenario: design reviews this week. Two events dominate, plus the prep
  // email and the exploration doc.
  // ---------------------------------------------------------------------------
  {
    id: "q2",
    trigger: "design review",
    answer:
      "You have two design reviews this week: checkout redesign on Thursday at 3pm " +
      "and mobile nav patterns on Friday at 11am. Elena shared the v3 checkout mocks " +
      "yesterday evening; Marcus has been pushing for a simpler progress indicator. " +
      "Raj posted the Friday agenda this morning.",
    results: [
      {
        id: "q2-r1",
        source: "Calendar",
        title: "Design Review — Checkout Redesign",
        subtitle: "Thu Apr 24 · 3:00–4:00 PM · Design studio / Zoom",
        from: "Marcus Jackson (organizer)",
        date: "Apr 24, 2026 · 3:00 PM",
        body:
          "Review of the v3 checkout redesign ahead of engineering hand-off next " +
          "week. Elena will walk through the updated flow; Marcus wants focused " +
          "discussion on the progress indicator (leaning toward a single-line " +
          "variant) and the Apple Pay surface. Raj joining to flag platform " +
          "constraints on the iOS side. Pre-read attached: checkout_v3.fig.",
        highlights: ["Design Review", "checkout", "progress indicator"],
        openInApp: "Calendar",
      },
      {
        id: "q2-r2",
        source: "Calendar",
        title: "Design Review — Mobile Nav Patterns",
        subtitle: "Fri Apr 25 · 11:00–12:00 PM · Design studio",
        from: "Raj Patel (organizer)",
        date: "Apr 25, 2026 · 11:00 AM",
        body:
          "Exploration review for the mobile nav patterns Elena has been " +
          "prototyping. Goal: narrow from five options to two that Elena " +
          "can put into user testing next week. Raj to share the research " +
          "brief at the start. Not a decision meeting — just alignment on " +
          "which directions are worth testing. Agenda posted in #design-review.",
        highlights: ["Design Review", "mobile nav", "user testing"],
        openInApp: "Calendar",
      },
      {
        id: "q2-r3",
        source: "Mail",
        title: "Re: Checkout v3 mocks ready for review",
        subtitle:
          "Elena Volkov — final tweaks are in, including Marcus's progress-indicator note…",
        from: "Elena Volkov <elena@northwind.io>",
        date: "Apr 23, 2026 · 6:12 PM",
        body:
          "Sending the v3 mocks ahead of tomorrow's design review. I've " +
          "incorporated Marcus's progress-indicator simplification (single " +
          "line, no step numbers) and the Apple Pay surface Raj asked about. " +
          "One open question I'd love input on: should the confirm button be " +
          "sticky to the bottom of the screen on mobile, or inline? Leaning " +
          "sticky but happy to be argued out of it. File attached.",
        highlights: ["design review", "checkout", "progress indicator"],
        openInApp: "Mail",
      },
      {
        id: "q2-r4",
        source: "Documents",
        title: "Mobile_Nav_Exploration.fig",
        subtitle: "Elena Volkov · Figma · modified Apr 22",
        from: "Elena Volkov",
        date: "Apr 22, 2026 · 4:58 PM",
        body:
          "Five candidate directions for the mobile nav refresh. Tabs: " +
          "Option A (bottom tab bar, simplified), Option B (edge swipe), " +
          "Option C (hamburger + bottom CTA), Option D (pull-down drawer), " +
          "Option E (persistent side sheet). Research brief from Raj lives " +
          "in the first frame. Intended as the starting point for Friday's " +
          "design review; goal is to pick two to take into user testing.",
        highlights: ["mobile nav", "design review", "user testing"],
        openInApp: "Figma",
      },
    ],
  },

  // ---------------------------------------------------------------------------
  // Query 3 — MIXED DOCUMENTS + iMESSAGE.
  // Scenario: onboarding a new engineer next week. Info lives across two
  // documents, a text thread, and an HR email.
  // ---------------------------------------------------------------------------
  {
    id: "q3",
    trigger: "onboarding",
    answer:
      "David starts Monday. Aisha finalized the two-week plan with shadowing " +
      "rotations through backend (week 1) and infra (week 2). Omar flagged two " +
      "outstanding access issues: David still needs a GitHub org invite and a VPN " +
      "certificate. HR confirmed his laptop shipped Friday.",
    results: [
      {
        id: "q3-r1",
        source: "Documents",
        title: "David_Kim_Onboarding_Plan.docx",
        subtitle: "Aisha Williams · ~/Documents/Hiring/2026 · modified Apr 20",
        from: "Aisha Williams",
        date: "Apr 20, 2026 · 3:14 PM",
        body:
          "Two-week onboarding plan for David Kim (starting Apr 28). Week 1 " +
          "is backend rotation — shadowing Omar on the payments service, " +
          "pairing with Priya on Tuesday, first small fix expected by Friday. " +
          "Week 2 shifts to infra: on-call shadowing, a half-day session with " +
          "the SRE team, and his first ticket on the deployment pipeline. " +
          "Expectation-setting 1:1 with Aisha end of week 2.",
        highlights: ["onboarding", "David Kim", "backend", "infra"],
        openInApp: "Pages",
      },
      {
        id: "q3-r2",
        source: "Documents",
        title: "Engineering_Onboarding_Checklist.md",
        subtitle: "Omar Farouk · ~/eng-handbook · modified Apr 10",
        from: "Omar Farouk",
        date: "Apr 10, 2026 · 10:47 AM",
        body:
          "Generic engineering onboarding checklist. Pre-start: laptop " +
          "provisioned, GitHub org invite sent, VPN cert issued, Slack " +
          "channels joined, 1:1 calendar invites placed. Day one: dev env " +
          "setup, first PR (README typo fix is fine), intro rounds. Week " +
          "one: shadowing assignment, reading list, first small fix. Owner " +
          "for keeping this current: whoever ran the last onboarding.",
        highlights: ["onboarding", "GitHub", "VPN"],
        openInApp: "Code",
      },
      {
        id: "q3-r3",
        source: "Messages",
        title: "Aisha Williams",
        subtitle:
          "heads up — David still doesn't have the github org invite, omar's on it…",
        from: "Aisha Williams",
        date: "Apr 23, 2026 · 4:22 PM",
        body:
          "heads up — David still doesn't have the github org invite, omar's " +
          "on it but wanted to flag in case it drags past Friday. also the " +
          "VPN cert request is stuck in IT's queue, omar pinged them twice. " +
          "if these aren't resolved by Monday morning we'll have to punt the " +
          "first-day PR plan. I'll DM you if it slips.",
        highlights: ["David", "github org invite", "VPN"],
        openInApp: "Messages",
      },
      {
        id: "q3-r4",
        source: "Mail",
        title: "Laptop shipped for David Kim — tracking #1Z999AA10",
        subtitle:
          "HR — David's MacBook Pro shipped Friday Apr 17, expected delivery Apr 24…",
        from: "HR <hr@northwind.io>",
        date: "Apr 19, 2026 · 10:05 AM",
        body:
          "Confirming David Kim's 16-inch MacBook Pro shipped Friday Apr 17 " +
          "via UPS Next Day Air, tracking #1Z999AA10. Expected delivery " +
          "Thursday Apr 23 or Friday Apr 24. Device has been pre-enrolled in " +
          "MDM; keyboard cover and Bluetooth mouse included. Please confirm " +
          "receipt when it arrives so IT can release the VPN cert.",
        highlights: ["David Kim", "laptop", "onboarding"],
        openInApp: "Mail",
      },
    ],
  },
];

// =============================================================================
// Helpers
// =============================================================================

/**
 * Find the MockQuery whose trigger phrase is a case-insensitive SUBSTRING
 * match against the user's typed input.
 *
 * Matching direction: we check whether the trigger CONTAINS the input
 * (`trigger.includes(input)`), so a short query like "bud" surfaces the
 * full "q2 budget" trigger. This matches Spotlight's incremental-typing
 * behaviour: the shorter the input, the more likely to match something.
 *
 * Returns null for the empty string (nothing typed yet) so the caller can
 * easily gate the "show answer/results" state on a real input being present.
 *
 * Note on ambiguity: if two triggers matched the same input we'd want tie-
 * breaking, but in this prototype the three triggers ("q2 budget",
 * "design review", "onboarding") have no overlapping substrings of length
 * ≥3, so the first-match-wins behaviour here is fine.
 */
export function findMatchingQuery(input: string): MockQuery | null {
  const needle = input.trim().toLowerCase();
  if (needle.length === 0) return null;
  for (const q of mockQueries) {
    if (q.trigger.toLowerCase().includes(needle)) {
      return q;
    }
  }
  return null;
}
