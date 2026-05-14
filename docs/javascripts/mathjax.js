/* MathJax 3 + pymdownx Arithmatex (generic). Material: setup/extensions → Arithmatex. */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(function () {
    if (!window.MathJax || !window.MathJax.startup) return;
    window.MathJax.startup.output.clearCache();
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise();
  });
}
