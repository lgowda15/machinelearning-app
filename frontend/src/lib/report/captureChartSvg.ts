import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import type { ReactElement } from "react";
import { resolveCssVars } from "./resolveCssVars";

export interface CapturedSvg {
  element: SVGSVGElement;
  width: number;
  height: number;
}

/**
 * Renders a chart component off-screen at fixed pixel dimensions and
 * returns its rendered `<svg>` with every `var(--color-*)` already resolved
 * to a literal value. `flushSync` guarantees the DOM is fully painted
 * before the `<svg>` is read out of the detached container -- a plain
 * `root.render()` doesn't (docs/plans/pdf-report.md's `captureChartSvg`
 * design decision).
 */
export function captureChartSvg(element: ReactElement, width: number, height: number): CapturedSvg {
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.left = "-10000px";
  container.style.top = "0";
  container.style.width = `${width}px`;
  container.style.height = `${height}px`;
  document.body.appendChild(container);

  const root = createRoot(container);

  try {
    flushSync(() => {
      root.render(element);
    });

    const svg = container.querySelector("svg");
    if (!svg) {
      throw new Error("captureChartSvg: the chart element did not render an <svg>");
    }

    resolveCssVars(svg);

    return {
      element: svg,
      width: Number(svg.getAttribute("width")) || width,
      height: Number(svg.getAttribute("height")) || height,
    };
  } finally {
    root.unmount();
    container.remove();
  }
}
