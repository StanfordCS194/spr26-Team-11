// =============================================================================
// Atlas — ResultRow
// =============================================================================
//
// One row of the Top K results list. Owns its own DOM ref + useLayoutEffect
// so that when its `selected` prop becomes true, the list scrolls just
// enough to bring this row into view. Self-contained: parents only need to
// pass a result, a selected flag, and a click handler.
// =============================================================================

import { useLayoutEffect, useRef } from "react";
import type { MockResult } from "../lib/types";
import { shortDate, sourceChipModifier } from "../lib/format";

export function ResultRow({
  result,
  selected,
  onClick,
}: {
  result: MockResult;
  selected: boolean;
  onClick: () => void;
}) {
  // Class list: base "result-row" plus the selected modifier when active.
  // Inline template-string concat rather than a helper like classnames —
  // the prototype has no other use for it, so a dependency is overkill.
  const className = `result-row${selected ? " is-selected" : ""}`;

  // Chip modifier derives from the SourceType. Lowercased to match the CSS
  // conventions (`result-chip--mail`, etc.).
  const chipModifier = sourceChipModifier(result.source);

  // Keep the selected row visible inside the scrollable .results-list. When
  // the user arrows past the last visible row, the list scrolls just enough
  // to bring the new selection into view. `block: "nearest"` ensures we
  // never scroll a row that's already on screen — no jumpiness when the
  // selection moves between two visible rows.
  const rowRef = useRef<HTMLLIElement>(null);
  // useLayoutEffect (not useEffect) so the scroll happens BEFORE the browser
  // paints. Otherwise the new .is-selected highlight paints at the row's old
  // off-screen position, the user sees a blue flash, and only then does the
  // scroll catch up. With useLayoutEffect the className change and the scroll
  // adjustment land in the same visual frame.
  useLayoutEffect(() => {
    if (selected) {
      rowRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [selected]);

  return (
    <li
      ref={rowRef}
      className={className}
      role="option"
      aria-selected={selected}
      onClick={onClick}
    >
      <span className={`result-chip result-chip--${chipModifier}`}>
        {result.source}
      </span>
      <div className="result-row__main">
        {/* Title and snippet each truncate with ellipsis on overflow — see
            styles.css. Keeping the structure flat (no extra nesting)
            reduces the number of layout boundaries the observer has to
            measure on resize. */}
        <div className="result-row__title">{result.title}</div>
        <div className="result-row__snippet">{result.subtitle}</div>
      </div>
      <span className="result-row__date">{shortDate(result.date)}</span>
    </li>
  );
}
