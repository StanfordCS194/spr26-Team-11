// =============================================================================
// Atlas — Renderer Entry Point
// =============================================================================
//
// This file is the very first thing the React side of the app runs. It is
// loaded by index.html via <script type="module" src="/src/main.tsx">.
//
// Responsibilities — kept deliberately tiny:
//   1. Find the #root mount point in the DOM.
//   2. Create a React 18 root and render <App /> into it.
//   3. Pull in the global stylesheet so our CSS is in the bundle.
//
// Anything UI-related belongs in App.tsx (or its children), not here. Keeping
// this file boring means we rarely need to touch it as the project grows.
// =============================================================================

import React from "react";
import ReactDOM from "react-dom/client";

// Global stylesheet. Vite picks this import up at build time and inlines/links
// it into index.html. We use plain CSS (no preprocessor, no CSS-in-JS) per the
// project's stack decision — fewer moving parts for a prototype.
import "./styles.css";

import App from "./App";

// Find the mount node defined in index.html. The "!" tells TypeScript we know
// the element exists — it's a hard-coded part of index.html and the alternative
// (a runtime null check) would just throw a less useful error than React's own
// missing-root error if we ever broke the HTML.
const rootElement = document.getElementById("root")!;

// React 18's createRoot enables concurrent features. We don't lean on them yet
// but using the modern API now means we don't have to migrate later.
ReactDOM.createRoot(rootElement).render(
  // StrictMode double-invokes effects in dev to surface bugs early. It has
  // zero effect in production builds, so it's pure upside during prototyping.
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
