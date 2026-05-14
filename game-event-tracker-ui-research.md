# Game Activity / Event Tracker Dashboard — UI Patterns Research

## Context
A hardcore mobile game player needs a beautiful UI for a game activity tracker that shows:
- Upcoming activities with start/end dates
- Version countdown timers
- Stream preview times
- Across **multiple games**

Reference: Notion calendar, Steam wishlist, Genshin wiki, Discord events.

---

## Reference App Analysis

### 1. WuWa Tracker (wuwatracker.com/timeline)
**Type:** Vertical timeline view for Wuthering Waves
- Clean horizontal timeline strip at top showing date markers
- Below it: event cards stacked vertically, organized by date
- Each card shows: event name, start/end dates, icon, category tag
- Color-coded by event type (banner, in-game event, tower, etc.)
- **Key pattern:** Timeline spine + event cards beneath, timezone switcher
- Mobile: vertical scroll, cards stack naturally

### 2. Paimon.moe (Genshin Impact Timeline)
**Type:** Horizontal timeline for Genshin events
- Horizontal scrolling timeline with date nodes
- Event items placed along the timeline with connectors
- Banner/wish events distinguished from in-game events via color
- Shows version patches as major dividers
- Countdown to next event visible without clicking
- **Key pattern:** Patch-based grouping, horizontal scroll, visual event connectors

### 3. Notion Dashboards
**Type:** Widget-based dashboard
- Drag-and-drop widgets: calendars, tables, timelines, charts
- Calendar view shows events as blocks on date grid
- Timeline view shows events as horizontal bars (Gantt-style)
- **Key pattern:** Flexible widget layout, multiple views of same data
- Filter by property, sort, group options per widget

### 4. Steam Wishlist / Personal Calendar
**Type:** Card grid + calendar
- Game cards with: cover art, title, release date, price, wishlist button
- Personal Calendar groups wishlisted games by release week
- Countdown badges on cards ("Releases in X days")
- **Key pattern:** Visual-first cards, countdown badges, week-grouped calendar

### 5. Discord Events (Apollo bot)
**Type:** Calendar + list hybrid
- Calendar strip at top, list of events below
- Event cards show: event name, date/time, server, attendee count
- "Interested/Going" toggle on each event
- **Key pattern:** RSVP affordance, attendee count, list+calendar toggle

---

## Key UI Patterns for Multi-Game Event Tracker

### A. Timeline View (Best for: event sequences, version counts)
- **Horizontal scrolling timeline** with date/time markers
- Events displayed as nodes or bars along the timeline
- Zoom levels: day / week / patch / month
- Now indicator ("now" line) to show current time position
- Multiple rows per game to avoid overlap
- **Variant:** Vertical timeline (WuWa style) — better for mobile vertical scroll

### B. Calendar View (Best for: date-at-a-glance, scheduling)
- Month/week/day grid
- Events as colored blocks on dates
- Click date to expand all events for that day
- Game color-coding lets user scan columns
- **Variant:** Agenda list below calendar (Notion style)

### C. Card-Based Dashboard (Best for: overview + detail)
- Grid of game cards at top (each game as a column)
- Below: upcoming events for selected game
- Cards show: game icon, event name, countdown, status badge
- Compact mode with just next event per game
- Expanded mode with full event list

### D. Countdown Widgets (Best for: urgency, FOMO)
- Prominent countdown timers for: next event, version release, stream start
- Circular progress or digital clock format
- Multiple countdowns visible simultaneously
- "Starting soon" state with visual urgency (pulsing, color shift)
- **Mobile pattern:** Countdown hero section at top, scrollable list below

### E. Hybrid Layout (Most Recommended)
```
┌──────────────────────────────────────────────┐
│  Countdown Hero Strip (next 3 events)        │
├──────────────────────────────────────────────┤
│  [Game A]  [Game B]  [Game C]  [+Add]        │  ← Game tabs/chips
├──────────────────────────────────────────────┤
│  Timeline View  │  Calendar  │  List         │  ← View toggle
├──────────────────────────────────────────────┤
│                                              │
│  Main content area (changes per view)        │
│  - Timeline: horizontal event bars           │
│  - Calendar: date grid with event blocks     │
│  - List: grouped event cards                 │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Game-Specific UX Considerations

1. **Timezone support** — Must show local time; games have different server times (UTC+8 for Genshin, various for others). WuWa tracker exemplifies this well.

2. **Event types** — Differentiate:
   - In-game events (limited time)
   - Banner/gacha pulls
   - Version updates (maintenance + launch)
   - Developer streams
   - Real-world events (esports)

3. **Recurring events** — Daily tasks / weekly resets need different treatment from one-time events

4. **Multiple games = multi-column** — Each game gets a "lane" or filter chip; shared calendar overlays events from all selected games

5. **Notification priming** — Show "X hours until event" prominently to drive action

---

## Mobile-First Considerations

- **Vertical scroll** favors: card list, vertical timeline
- **Horizontal scroll** works in: tabbed sub-sections, timeline zoom
- **Bottom navigation** for main sections (Home/Timeline/Calendar/Settings)
- **Pull-to-refresh** for live countdown updates
- **Sticky headers** for current date / game filter

---

## Color & Visual Language

| Element | Recommendation |
|---|---|
| Game cards | Dark background, game brand color accent, cover art |
| Countdown timers | High contrast, monospace or digital font, accent color pulse |
| Event categories | Consistent color per type across all games |
| Now indicator | Bright accent line/dot with "LIVE" or "NOW" badge |
| Empty states | Friendly illustration + "No upcoming events" + next event CTA |
| Urgent (starting soon) | Pulsing border, warmer accent color |

---

## Typography & Spacing

- **Countdowns:** Large (32-48px), bold, tabular numbers
- **Event names:** Medium (16-18px), semi-bold
- **Dates/times:** Small (12-14px), muted color, relative ("in 2h") + absolute ("May 15, 10:00 AM")
- **Card padding:** 12-16px, rounded corners (8-12px)
- **Grid gap:** 12-16px between cards

---

## Interaction Patterns

1. **Tap event card** → Expand inline or open detail sheet (slide up on mobile)
2. **Long-press game chip** → Quick filter to that game only
3. **Swipe timeline** → Navigate days/weeks
4. **Tap countdown** → Set reminder / add to device calendar
5. **Toggle view** → Smooth transition between timeline/calendar/list

---

## Inspiration Sources
- WuWa Tracker timeline: https://wuwatracker.com/timeline
- Paimon.moe Genshin timeline: https://paimon.moe/timeline
- Notion Dashboards: https://www.notion.com/nb/help/dashboards
- Steam Labs Personal Calendar
- Apollo Discord Events: https://apollo.fyi/
