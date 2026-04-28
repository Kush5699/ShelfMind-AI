# Design System Specification: ShelfMind AI

## 1. Overview & Creative North Star: "The Neural Architect"
This design system is built to transform complex retail data into a high-fidelity, editorial experience. We are moving away from the "generic SaaS dashboard" and toward **The Neural Architect**—a creative North Star that treats data intelligence as a premium, living entity.

The aesthetic profile leans into "Dark Editorial Glassmorphism." By utilizing intentional asymmetry, deep tonal layering, and high-contrast typography, we create an interface that feels less like a tool and more like an authoritative intelligence partner. We prioritize breathing room over information density, ensuring every metric feels earned and every insight feels profound.

---

## 2. Color & Surface Architecture

### Palette Definition
We utilize a sophisticated spectrum of teals and purples, grounded in a deep, obsidian-space foundation.

*   **Primary (Teal):** `#46f1c5` (Surface) | `#00d4aa` (Container). Use for core actions and "success" retail growth metrics.
*   **Secondary (Purple):** `#c8bfff` (Surface) | `#442bb5` (Container). Use for AI-generated insights and predictive logic.
*   **Tertiary (Gold/Amber):** `#ffcea6`. Reserved for high-value SKU alerts or inventory warnings.
*   **Neutral (Surface):** `#12131a`. The foundation for all layouts.

### The "No-Line" Rule
To maintain a premium, seamless feel, **1px solid borders are prohibited for sectioning.** 
*   **Boundaries:** Define spatial transitions solely through background shifts. For example, a `surface-container-low` panel should sit directly on a `surface` background without a stroke.
*   **Visual Polish:** Use the `surface-container` tiers (Lowest to Highest) to create "nested" depth.

### Glass & Gradient Rule
Floating elements (modals, dropdowns, hovered cards) must utilize the **Glassmorphism Spec**:
*   **Background:** `rgba(255, 255, 255, 0.03)`
*   **Blur:** `16px` backdrop-filter.
*   **Ghost Border:** `1px solid rgba(255, 255, 255, 0.06)` (The only exception to the No-Line rule).

---

## 3. Typography: Editorial Authority
We use **Inter** as a variable font to create a hierarchy that feels like a high-end financial journal.

*   **Display (lg/md):** 3.5rem - 2.75rem | Bold (700-800). Used for primary retail KPIs (e.g., Total Revenue ₹).
*   **Headline (sm/md):** 1.5rem - 1.75rem | Semi-Bold (600). Used for section titles.
*   **Metric Gradients:** All primary numerical metrics must use a linear gradient from `#00d4aa` to `#00b4d8`.
*   **Body (md/lg):** 0.875rem - 1rem | Regular (400). Optimized for readability of SKU descriptions and inventory logs.

---

## 4. Elevation & Depth: Tonal Layering
Traditional shadows are too "heavy" for this system. We use light and transparency to convey height.

*   **The Layering Principle:** Depth is achieved by "stacking." 
    *   *Base:* `surface`
    *   *Section:* `surface-container-low`
    *   *Card:* `surface-container`
*   **Ambient Shadows:** For "floating" AI insights, use an extra-diffused glow: `0 20px 40px rgba(0, 0, 0, 0.4)` combined with a tinted teal bloom: `0 4px 30px rgba(0, 212, 170, 0.15)`.
*   **Interactions:** On hover, elements should execute a `translateY(-4px)` movement with a `cubic-bezier(0.2, 0.8, 0.2, 1)` transition for a "magnetic" feel.

---

## 5. Component Strategies

### Buttons & CTAs
*   **Primary:** A gradient fill (`primary` to `primary-container`) with white text. No border.
*   **Secondary:** Ghost style. Transparent background with the "Ghost Border" (outline-variant at 20% opacity).
*   **Tertiary:** Text-only with a `primary-fixed` color and an underline that appears only on hover.

### Intelligence Cards
*   **Constraint:** Never use divider lines. 
*   **Grouping:** Use vertical whitespace (referencing the `xl` 1.5rem spacing scale) to separate SKU data from predictive charts.
*   **Radii:** All cards must use `24px` (`xl`) corner rounding to soften the industrial data.

### Input Fields
*   **Style:** Minimalist. Only a bottom border (outline-variant) that glows to `primary` on focus.
*   **Labels:** Use `label-sm` (uppercase) with `0.05em` letter spacing for an "architectural" look.

### Contextual Retail Elements (Custom)
*   **Currency Display:** Indian Rupee (₹) symbols should be set in a slightly lower weight than the value to emphasize the number itself.
*   **Stock Indicators:** Use "Pulse" dots instead of text badges (Green pulse for "In Stock," Red pulse for "Stockout").

---

## 6. Do’s and Don’ts

### Do:
*   **Do** use asymmetrical layouts. A 2/3 and 1/3 column split is preferred over a centered grid.
*   **Do** allow the background gradient (`#06070d` to `#0d1b2a`) to bleed through glass elements.
*   **Do** use `primary-container` for subtle "glow" backgrounds behind iconography.

### Don’t:
*   **Don’t** use pure black (`#000000`) or pure white for text. Use `on-surface` (`#e3e1eb`) for maximum premium legibility.
*   **Don’t** use standard 4px or 8px radiuses. It makes the platform look like a legacy enterprise tool. Stick to `12px` (md) and `24px` (xl).
*   **Don’t** use "Information Density" as an excuse for clutter. If a screen feels full, increase the `surface` padding.