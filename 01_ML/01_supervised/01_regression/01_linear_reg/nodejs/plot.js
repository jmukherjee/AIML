import { writeFileSync } from 'node:fs';

export function writePlotlyHtml(filePath, title, traces, layout = {}) {
  const fullLayout = { title, ...layout };
  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>${title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="chart" style="width:100%;height:90vh;"></div>
  <script>
    Plotly.newPlot('chart', ${JSON.stringify(traces)}, ${JSON.stringify(fullLayout)});
  </script>
</body>
</html>`;
  writeFileSync(filePath, html);
}
