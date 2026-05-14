/* MathJax 3 + pymdownx Arithmatex (generic). Official Arithmatex generic snippet:
   https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/#loading-mathjax
   Use ignoreHtmlClass ".*" (not ".*|") so processing is limited to .arithmatex correctly. */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*",
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
